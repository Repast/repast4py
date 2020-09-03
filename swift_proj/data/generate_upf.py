"""Generate UPF for the JCCM Model"""

import json

nodes = 1
ppn = 36
replicates = 30

humans = 1000
zombies = 50
world_size = 20
stop_at = 100

seed = 0
run = 1
lines = []


for r in range(0,replicates):
    seed += 1

    param = {}
    param['run'] = run
    param['random.seed'] = seed
    param['stop.at'] = stop_at
    param['human.count'] = humans
    param['zombie.count'] = zombies
    param['world.width'] = world_size
    param['world.height'] = world_size

    lines.append(json.dumps(param) + '\n')

    run += 1

with open('upf_01.txt', 'w') as f:
    f.writelines(lines)

