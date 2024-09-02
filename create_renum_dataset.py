import os
import json
import random
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Simple example of argparse")

    # 添加参数
    parser.add_argument("--base", type=int, default=1000)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--outputs', type=str, required=True)

    # 解析参数
    return parser.parse_args()

def add(i1, i2):
    # return float(format(i1 + i2, ".5f"))
    return f"{i1+i2:.5f}".rstrip("0").rstrip(".")

def minus(i1, i2):
    # return float(format(i1 - i2, ".5f"))
    return f"{i1-i2:.5f}".rstrip("0").rstrip(".")

def mul(i1, i2):
    # return float(format(i1 * i2, ".8f"))
    return f"{i1*i2:.8f}".rstrip("0").rstrip(".")

def div(i1, i2):
    # return float(format(i1 / i2, ".8f"))
    return f"{i1/i2:.8f}".rstrip("0").rstrip(".")

def reversen(i):
    o = ''.join(reversed(str(i)))
    if o[-1] == '-':
        o = f"-{o[:-1]}"
    return o

def main():
    args = parse_args()
    
    t_data, e_data, rt_data, re_data = [], [], [], []
    n1t = np.zeros([args.base * 2, args.base * 2], dtype=int)
    n2t = np.zeros([args.base * 2, args.base * 2], dtype=int)
    # opw = [add, minus, mul, div]
    # ops = ['+','-', '*', '/']
    # opw = [add, minus, mul]
    # ops = ['+', '-', '*']
    opw = [div]
    ops = ['/']

    cnt = 0
    pbar = tqdm(total=args.samples, dynamic_ncols=True)
    while True:
        i1 = random.randint(-args.base, args.base - 1)
        f1 = random.randint(-args.base, args.base - 1)
        i2 = random.randint(-args.base, args.base - 1)
        f2 = random.randint(-args.base, args.base - 1)
        if n1t[i1, f1] == 0 or n2t[i2, f2] == 0:
            n1 = float(i1) + float(f1) / float(args.base)
            n2 = float(i2) + float(f2) / float(args.base)
            for ow, op in zip(opw, ops):
                if random.random() < 0.95:
                    if random.random() < 0.5:
                        if n2 != 0 or ow != div:
                            cal = ow(n1, n2)
                            t_data.append({
                                "i":f"{n1}{op}{n2}=",
                                "o":f"{cal}",
                            })
                            rt_data.append({
                                "i":f"{reversen(n1)}{op}{reversen(n2)}=",
                                "o":f"{reversen(cal)}",
                            })
                    else:
                        if n1 != 0 or ow != div:
                            cal = ow(n2, n1)
                            t_data.append({
                                "i":f"{n2}{op}{n1}=",
                                "o":f"{cal}",
                            })
                            rt_data.append({
                                "i":f"{reversen(n2)}{op}{reversen(n1)}=",
                                "o":f"{reversen(cal)}",
                            })
                else:
                    if random.random() < 0.5:
                        if n2 != 0 or ow != div:
                            cal = ow(n1, n2)
                            e_data.append({
                                "i":f"{n1}{op}{n2}=",
                                "o":f"{cal}",
                            })
                            re_data.append({
                                "i":f"{reversen(n1)}{op}{reversen(n2)}=",
                                "o":f"{reversen(cal)}",
                            })
                    else:
                        if n1 != 0 or ow != div:
                            cal = ow(n2, n1)
                            e_data.append({
                                "i":f"{n2}{op}{n1}=",
                                "o":f"{cal}",
                            })
                            re_data.append({
                                "i":f"{reversen(n2)}{op}{reversen(n1)}=",
                                "o":f"{reversen(cal)}",
                            })
            n2t[i2, f2] = n1t[i1, f1] == 1
            n1t[i1, f1] = 1
            pbar.update(1)
            cnt += 1
        if cnt == args.samples:
            break
    pbar.close()

    if not os.path.exists(args.outputs):
        os.makedirs(os.path.join(args.outputs, 's/train'))
        os.makedirs(os.path.join(args.outputs, 's/eval'))
        os.makedirs(os.path.join(args.outputs, 'r/train'))
        os.makedirs(os.path.join(args.outputs, 'r/eval'))

    with open(os.path.join(args.outputs, "s/train/data.jsonl"), 'w') as f:
        for item in t_data:
            f.write(json.dumps(item)+"\n")
    with open(os.path.join(args.outputs, "s/eval/data.jsonl"), 'w') as f:
        for item in e_data:
            f.write(json.dumps(item)+"\n")

    with open(os.path.join(args.outputs, "r/train/data.jsonl"), 'w') as f:
        for item in rt_data:
            f.write(json.dumps(item)+"\n")
    with open(os.path.join(args.outputs, "r/eval/data.jsonl"), 'w') as f:
        for item in re_data:
            f.write(json.dumps(item)+"\n")

if __name__ == "__main__":
    main()
