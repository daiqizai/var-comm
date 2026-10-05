#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct Trellis {
    int memory;
    int states;
    std::vector<double> half_symbols;
    std::vector<double> bit_symbols;

    Trellis(int memory_value, int first_generator, int second_generator)
        : memory(memory_value), states(1 << memory_value),
          half_symbols(states * states * 2 * memory), bit_symbols(states * 4) {
        for (int state = 0; state < states; ++state) {
            for (int bit = 0; bit < 2; ++bit) {
                const unsigned register_value = (state << 1) | bit;
                bit_symbols[state * 4 + bit * 2] = 1 - 2 * (__builtin_popcount(register_value & first_generator) & 1);
                bit_symbols[state * 4 + bit * 2 + 1] = 1 - 2 * (__builtin_popcount(register_value & second_generator) & 1);
            }
        }
        for (int state = 0; state < states; ++state) {
            for (int half = 0; half < states; ++half) {
                int current = state;
                for (int step = 0; step < memory; ++step) {
                    const int bit = (half >> (memory - 1 - step)) & 1;
                    for (int coordinate = 0; coordinate < 2; ++coordinate) {
                        half_symbols[((state * states + half) * memory + step) * 2 + coordinate]
                            = bit_symbols[current * 4 + bit * 2 + coordinate];
                    }
                    current = ((current << 1) | bit) & (states - 1);
                }
            }
        }
    }
};

const Trellis& get_trellis(int memory) {
    static const Trellis toy(2, 7, 5);
    static const Trellis standard(6, 0171, 0133);
    return memory == 2 ? toy : standard;
}

extern "C" int decode_token_map(
    int memory, int token_count, int information_bits, int final_state,
    const double* evidence, const double* log_prior, int prior_stride,
    uint8_t* decoded, double* score) {
    try {
        if ((memory != 2 && memory != 6) || token_count < 0 || information_bits < token_count * 2 * memory) return 1;
        const Trellis& trellis = get_trellis(memory);
        const int states = trellis.states;
        if (final_state < -1 || final_state >= states) return 2;
        const int tail_length = information_bits - token_count * 2 * memory;
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> metrics(states, infinity), next_metrics(states), middle(states);
        std::vector<double> first_cost(states * states), second_cost(states * states);
        std::vector<int> middle_previous(states);
        std::vector<uint8_t> token_previous(token_count * states), high_half(token_count * states);
        std::vector<uint8_t> bit_previous(tail_length * states);
        metrics[0] = 0;
        double removed = 0;
        for (int token = 0; token < token_count; ++token) {
            const double* first_evidence = evidence + token * 4 * memory;
            const double* second_evidence = first_evidence + 2 * memory;
            for (int pair = 0; pair < states * states; ++pair) {
                double first = 0;
                double second = 0;
                for (int coordinate = 0; coordinate < 2 * memory; ++coordinate) {
                    const double symbol = trellis.half_symbols[pair * 2 * memory + coordinate];
                    first -= first_evidence[coordinate] * symbol;
                    second -= second_evidence[coordinate] * symbol;
                }
                first_cost[pair] = first;
                second_cost[pair] = second;
            }
            for (int high = 0; high < states; ++high) {
                double best = infinity;
                int previous = 0;
                for (int state = 0; state < states; ++state) {
                    const double candidate = metrics[state] + first_cost[state * states + high];
                    if (candidate < best) { best = candidate; previous = state; }
                }
                middle[high] = best;
                middle_previous[high] = previous;
            }
            const double* prior = log_prior + token * prior_stride;
            for (int low = 0; low < states; ++low) {
                double best = infinity;
                int chosen_high = 0;
                for (int high = 0; high < states; ++high) {
                    const double candidate = middle[high] + second_cost[high * states + low] - prior[high * states + low];
                    if (candidate < best) { best = candidate; chosen_high = high; }
                }
                next_metrics[low] = best;
                token_previous[token * states + low] = middle_previous[chosen_high];
                high_half[token * states + low] = chosen_high;
            }
            const double minimum = *std::min_element(next_metrics.begin(), next_metrics.end());
            removed += minimum;
            for (int state = 0; state < states; ++state) metrics[state] = next_metrics[state] - minimum;
        }
        for (int step = 0; step < tail_length; ++step) {
            const double* current_evidence = evidence + (token_count * 2 * memory + step) * 2;
            for (int state = 0; state < states; ++state) {
                const int bit = state & 1;
                const int previous_zero = state >> 1;
                const int previous_one = previous_zero | (states >> 1);
                const double first = metrics[previous_zero]
                    - current_evidence[0] * trellis.bit_symbols[previous_zero * 4 + bit * 2]
                    - current_evidence[1] * trellis.bit_symbols[previous_zero * 4 + bit * 2 + 1];
                const double second = metrics[previous_one]
                    - current_evidence[0] * trellis.bit_symbols[previous_one * 4 + bit * 2]
                    - current_evidence[1] * trellis.bit_symbols[previous_one * 4 + bit * 2 + 1];
                const bool choose_one = second < first;
                next_metrics[state] = choose_one ? second : first;
                bit_previous[step * states + state] = choose_one ? previous_one : previous_zero;
            }
            const double minimum = *std::min_element(next_metrics.begin(), next_metrics.end());
            removed += minimum;
            for (int state = 0; state < states; ++state) metrics[state] = next_metrics[state] - minimum;
        }
        int current = final_state;
        if (current == -1) current = std::min_element(metrics.begin(), metrics.end()) - metrics.begin();
        *score = removed + metrics[current];
        for (int step = tail_length - 1; step >= 0; --step) {
            decoded[token_count * 2 * memory + step] = current & 1;
            current = bit_previous[step * states + current];
        }
        for (int token = token_count - 1; token >= 0; --token) {
            const int high = high_half[token * states + current];
            const int value = high * states + current;
            const int previous = token_previous[token * states + current];
            for (int bit = 0; bit < 2 * memory; ++bit) decoded[token * 2 * memory + bit] = (value >> (2 * memory - 1 - bit)) & 1;
            current = previous;
        }
        return current == 0 && std::isfinite(*score) ? 0 : 3;
    } catch (...) { return 4; }
}
