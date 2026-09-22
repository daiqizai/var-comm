#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Counters {
    uint64_t products = 0, backward = 0, deviations = 0, pushes = 0, pops = 0;
    uint64_t comparisons = 0, traceback = 0, peak_heap = 0, peak_bytes = 0;
};

struct Layer {
    int width, offset, choices;
    std::vector<double> costs;
};

struct Graph {
    int memory, states, information_bits;
    std::vector<Layer> layers;
    std::vector<double> distance;
    std::vector<uint8_t> policy;
    Counters& counters;
    uint64_t bytes = 0;

    Graph(int memory_value, int tokens, int bits, int terminal, const double* evidence,
          const double* prior, Counters& counts)
        : memory(memory_value), states(1 << memory_value), information_bits(bits), counters(counts) {
        const int generators[2] = {memory == 6 ? 0171 : 07, memory == 6 ? 0133 : 05};
        int offset = 0;
        while (offset < bits) {
            const bool token_part = offset < tokens * 2 * memory;
            const int width = token_part ? memory : 1;
            Layer layer{width, offset, 1 << width, {}};
            layer.costs.resize(states * layer.choices);
            for (int state = 0; state < states; ++state) {
                for (int value = 0; value < layer.choices; ++value) {
                    int current = state;
                    double cost = 0.0;
                    for (int step = 0; step < width; ++step) {
                        const int bit = (value >> (width - 1 - step)) & 1;
                        const int combined = (current << 1) | bit;
                        for (int output = 0; output < 2; ++output) {
                            const int sign = 1 - 2 * __builtin_parity(static_cast<unsigned>(combined & generators[output]));
                            cost -= sign * evidence[2 * (offset + step) + output];
                            ++counters.products;
                        }
                        current = combined & (states - 1);
                    }
                    if (token_part && offset % (2 * memory) == memory) {
                        cost -= prior[(offset / (2 * memory)) * states * states + state * states + value];
                    }
                    layer.costs[state * layer.choices + value] = cost;
                }
            }
            bytes += layer.costs.capacity() * sizeof(double) + sizeof(Layer);
            layers.push_back(std::move(layer));
            offset += width;
        }
        const int length = static_cast<int>(layers.size());
        const double infinity = std::numeric_limits<double>::infinity();
        distance.assign((length + 1) * states, infinity);
        policy.resize(length * states);
        for (int state = 0; state < states; ++state) {
            if (terminal < 0 || state == terminal) distance[length * states + state] = 0.0;
        }
        for (int position = length - 1; position >= 0; --position) {
            const Layer& layer = layers[position];
            for (int state = 0; state < states; ++state) {
                double best = infinity;
                int chosen = 0;
                for (int value = 0; value < layer.choices; ++value) {
                    const int next = ((state << layer.width) | value) & (states - 1);
                    const double cost = layer.costs[state * layer.choices + value] + distance[(position + 1) * states + next];
                    ++counters.backward;
                    if (cost < best) { best = cost; chosen = next; }
                }
                distance[position * states + state] = best;
                policy[position * states + state] = static_cast<uint8_t>(chosen);
            }
        }
        bytes += distance.capacity() * sizeof(double) + policy.capacity();
    }

    double cost(int position, int state, int next) const {
        const Layer& layer = layers[position];
        return layer.costs[state * layer.choices + (next & (layer.choices - 1))];
    }
};

struct Path {
    std::vector<uint8_t> states;
    std::vector<double> costs;
    int partition_start;
};

struct Alternative {
    double cost;
    int parent, position, next;
    uint64_t order;
};

struct Compare {
    Counters* counters;
    bool operator()(const Alternative& left, const Alternative& right) const {
        ++counters->comparisons;
        return left.cost > right.cost || (left.cost == right.cost && left.order > right.order);
    }
};

uint16_t checksum(const uint8_t* bits, int length, uint16_t initial) {
    uint16_t value = initial;
    for (int index = 0; index < length; ++index) {
        const bool top = ((value >> 15) ^ bits[index]) & 1;
        value = static_cast<uint16_t>(value << 1);
        if (top) value ^= 0x1021;
    }
    return value;
}

extern "C" int enumerate_whole_paths(
    const int* settings, const double* evidence, const double* prior, double seconds,
    uint8_t* decoded, double* scores, uint8_t* accepts, double* statistics) {
    const auto started = Clock::now();
    try {
        const int memory = settings[0], tokens = settings[1], bits = settings[2];
        const int initial = settings[3], terminal = settings[4], stop_bits = settings[5];
        const int maximum = settings[6], payload = settings[7], crc_initial = settings[8];
        const bool stop_on_crc = settings[9] != 0;
        const size_t queue_limit = static_cast<size_t>(settings[10]);
        if ((memory != 2 && memory != 6) || tokens < 0 || bits < tokens * 2 * memory || maximum < 1 ||
            initial < 0 || initial >= (1 << memory) || terminal < -1 || terminal >= (1 << memory) ||
            stop_bits < 1 || stop_bits > bits || (payload >= 0 && payload + 16 + memory != stop_bits)) return -1;
        Counters counters;
        Graph graph(memory, tokens, bits, terminal, evidence, prior, counters);
        int stop = 0;
        while (stop < static_cast<int>(graph.layers.size()) && graph.layers[stop].offset < stop_bits) ++stop;
        if (graph.layers[stop - 1].offset + graph.layers[stop - 1].width != stop_bits) return -2;
        const double setup = std::chrono::duration<double>(Clock::now() - started).count();
        std::vector<Path> paths;
        std::vector<Alternative> heap;
        Compare compare{&counters};
        Alternative current{graph.distance[initial], -1, -1, initial, 0};
        uint64_t sequence = 1, path_bytes = 0;
        int returned = 0, reason = 0;
        while (returned < maximum) {
            if (!std::isfinite(current.cost)) { reason = 2; break; }
            Path path{{}, {}, current.position + 1};
            path.states.resize(stop + 1);
            path.costs.resize(stop + 1);
            path.states[0] = static_cast<uint8_t>(initial);
            path.costs[0] = 0.0;
            if (current.parent >= 0) {
                const Path& parent = paths[current.parent];
                std::copy(parent.states.begin(), parent.states.begin() + current.position + 1, path.states.begin());
                std::copy(parent.costs.begin(), parent.costs.begin() + current.position + 1, path.costs.begin());
                path.states[current.position + 1] = static_cast<uint8_t>(current.next);
                path.costs[current.position + 1] = path.costs[current.position] + graph.cost(current.position, path.states[current.position], current.next);
            }
            for (int position = path.partition_start; position < stop; ++position) {
                const int state = path.states[position];
                const int next = graph.policy[position * graph.states + state];
                path.states[position + 1] = static_cast<uint8_t>(next);
                path.costs[position + 1] = path.costs[position] + graph.cost(position, state, next);
                ++counters.traceback;
            }
            uint8_t* candidate = decoded + static_cast<size_t>(returned) * stop_bits;
            for (int position = 0; position < stop; ++position) {
                const Layer& layer = graph.layers[position];
                const int value = path.states[position + 1] & (layer.choices - 1);
                for (int index = 0; index < layer.width; ++index) {
                    candidate[layer.offset + index] = (value >> (layer.width - 1 - index)) & 1;
                }
            }
            scores[2 * returned] = path.costs[stop] + graph.distance[stop * graph.states + path.states[stop]];
            scores[2 * returned + 1] = path.costs[stop];
            bool accepted = false;
            if (payload >= 0) {
                uint16_t received_crc = 0;
                for (int index = payload; index < payload + 16; ++index) received_crc = (received_crc << 1) | candidate[index];
                accepted = checksum(candidate, payload, static_cast<uint16_t>(crc_initial)) == received_crc;
            }
            accepts[returned] = static_cast<uint8_t>(accepted);
            path_bytes += path.states.capacity() + path.costs.capacity() * sizeof(double) + sizeof(Path);
            paths.push_back(std::move(path));
            ++returned;
            counters.peak_bytes = std::max(counters.peak_bytes, graph.bytes + path_bytes + heap.capacity() * sizeof(Alternative));
            if (accepted && stop_on_crc) { reason = 1; break; }
            if (seconds >= 0 && std::chrono::duration<double>(Clock::now() - started).count() >= seconds) { reason = 3; break; }
            if (returned == maximum) break;
            const Path& parent = paths.back();
            bool limited = false;
            for (int position = parent.partition_start; position < stop && !limited; ++position) {
                const Layer& layer = graph.layers[position];
                const int state = parent.states[position];
                for (int value = 0; value < layer.choices; ++value) {
                    const int next = ((state << layer.width) | value) & (graph.states - 1);
                    if (next == parent.states[position + 1]) continue;
                    ++counters.deviations;
                    const double cost = parent.costs[position] + graph.cost(position, state, next)
                        + graph.distance[(position + 1) * graph.states + next];
                    if (!std::isfinite(cost)) continue;
                    if (heap.size() == queue_limit) { limited = true; break; }
                    heap.push_back({cost, returned - 1, position, next, sequence++});
                    std::push_heap(heap.begin(), heap.end(), compare);
                    ++counters.pushes;
                }
            }
            counters.peak_heap = std::max(counters.peak_heap, static_cast<uint64_t>(heap.size()));
            counters.peak_bytes = std::max(counters.peak_bytes, graph.bytes + path_bytes + heap.capacity() * sizeof(Alternative));
            if (limited) { reason = 4; break; }
            if (seconds >= 0 && std::chrono::duration<double>(Clock::now() - started).count() >= seconds) { reason = 3; break; }
            if (heap.empty()) { reason = 2; break; }
            std::pop_heap(heap.begin(), heap.end(), compare);
            current = heap.back();
            heap.pop_back();
            ++counters.pops;
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        const double values[] = {static_cast<double>(counters.products), static_cast<double>(counters.backward),
            static_cast<double>(counters.deviations), static_cast<double>(counters.pushes), static_cast<double>(counters.pops),
            static_cast<double>(counters.comparisons), static_cast<double>(counters.traceback), static_cast<double>(counters.peak_heap),
            static_cast<double>(counters.peak_bytes), setup, elapsed, static_cast<double>(reason), static_cast<double>(returned),
            seconds < 0 ? 0.0 : std::max(0.0, elapsed - seconds)};
        std::copy(std::begin(values), std::end(values), statistics);
        return returned;
    } catch (...) { return -99; }
}
