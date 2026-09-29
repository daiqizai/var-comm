from run_preflight import *
if __name__=='__main__':
 runner=delivery_chain.Runner()
 runner.run('gate2_calibration_crosscheck',['verify_gate2'],OUT/'gate2_crosscheck.json')
