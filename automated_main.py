import subprocess

commands = [
    'python train.py --weights "" --cfg yolov3-nano.yaml --data yperv2.yaml --normalize --imgsz 640 --setseed --seednum 123 --epochs 100 --batch-size 16 --adam --multi-scale',
    'python train.py --weights "" --cfg yolov3-nano.yaml --data yperv2.yaml --normalize --imgsz 960 --setseed --seednum 123 --epochs 100 --batch-size 16 --adam',
    'python train.py --weights "" --cfg yolov3-nano.yaml --data yperv2.yaml --normalize --imgsz 640 --setseed --seednum 123 --epochs 100 --batch-size 16 --adam',
]

for command in commands:
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
    )
    process.wait()

    print(process.returncode)
