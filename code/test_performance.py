import json
from agents import Performance_Evaluation_Agent
from copy import deepcopy

shiftable_devices = {
    1: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    2: ['Washing Machine', 'Dishwasher'],
    3: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    4: ['Washing Machine (1)', 'Washing Machine (2)'],
    5: ['Tumble Dryer'], # , 'Washing Machine' --> consumes energy constantly; , 'Dishwasher' --> noise at 3am
    6: ['Washing Machine', 'Dishwasher'],
    7: ['Tumble Dryer', 'Washing Machine', 'Dishwasher'],
    8: ['Washing Machine'], # 'Dryer' --> consumes constantly
    9: ['Washer Dryer', 'Washing Machine', 'Dishwasher'],
    10: ['Washing Machine'] #'Dishwasher'
}

active_appliances = {
    1: deepcopy(shiftable_devices[1]) + ['Television Site', 'Computer Site'],
    2: deepcopy(shiftable_devices[2]) + ['Television', 'Microwave', 'Toaster', 'Hi-Fi', 'Kettle'],
    3: deepcopy(shiftable_devices[3]) + ['Toaster', 'Television', 'Microwave', 'Kettle'],
    4: deepcopy(shiftable_devices[4]) + ['Television Site', 'Kettle'], #'Microwave', 'Computer Site' --> consume energy constantly
    5: deepcopy(shiftable_devices[5]) + ['Television Site', 'Combination Microwave', 'Kettle', 'Toaster'], # 'Computer Site', --> consumes energy constantly
    6: deepcopy(shiftable_devices[6]) + ['MJY Computer', 'Kettle', 'Toaster'], #, 'PGM Computer', 'Television Site' 'Microwave' --> consume energy constantly
    7: deepcopy(shiftable_devices[7]) + ['Television Site', 'Toaster', 'Kettle'],
    8: deepcopy(shiftable_devices[8]) + ['Toaster', 'Kettle'], # 'Television Site', 'Computer' --> consume energy constantly
    9: deepcopy(shiftable_devices[9]) + ['Microwave', 'Kettle'], #'Television Site', 'Hi-Fi' --> consume energy constantly
    10: deepcopy(shiftable_devices[10]) + ['Magimix (Blender)', 'Microwave'] # 'Television Site' --> consume energy constantly
}

thresholds = {
    1: 0.15,
    2: 0.01,
    3: 0.01,
    4: 0.01,
    5: 0.025,
    6: 0.065,
    7: 0.01,
    8: 0.01, # washing machine over night
    9: 0.01,
    10: 0.01
}

DATA_PATH = '../data/'
EXPORT_PATH = '../export/'

household_id = 3

config = {'data': {'household': deepcopy(household_id)}}
config['user_input'] = {
    'shiftable_devices': deepcopy(shiftable_devices[config['data']['household']]),
    'active_appliances': deepcopy(active_appliances[config['data']['household']]),
    'threshold': deepcopy(thresholds[config['data']['household']])
}


evaluation = Performance_Evaluation_Agent(DATA_PATH, config=config, model_type="xgboost", load_data=True, weather_sel=True, xai=True)
evaluation.get_default_config('preparation')
evaluation.pipeline('preparation')


evaluation.get_default_config(['activity', 'usage', 'load'])

evaluation.pipeline(['activity', 'usage', 'load'])
evaluation.pipeline(agents='recommendation', activity_threshold=0.625, usage_threshold=0.125)

print(evaluation.output['recommendation'])
evaluation.dump(EXPORT_PATH)
evaluation.get_agent_scores(learning_rate=0.1, max_depth=6, reg_lambda=1, reg_alpha=0, xai=True)
evaluation.dump(EXPORT_PATH)


