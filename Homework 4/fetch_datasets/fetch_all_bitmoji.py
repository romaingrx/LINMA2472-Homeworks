import os
import json
import requests
from tqdm import tqdm
from itertools import product

import multiprocessing as mp
from concurrent import futures

# ========SETUP=========

WORKERS = 4

GLOBAL_START = 300000000 - (50000) - (60)
HOW_MUCH_BITMOJI = max(5000, WORKERS)

REMOVE_TEXT_AND_BACKGROUND = True


# ======================


def getBitmoji(prod):
    id, template = prod
    did = f"./bitmoji/{id}"
    if REMOVE_TEXT_AND_BACKGROUND:
        res = requests.get(f"https://render.bitstrips.com/render/{template}/{id}_2-v1.png")
    else:
        res = requests.get(
            f"https://render.bitstrips.com/v2/cpanel/{template}-{id}_2-s1-v1.png?transparent=0&palette=1")  # &width=400
    if res.status_code == 200:
        if not os.path.exists(did): os.mkdir(did)
        file = open(f"{did}/{id}-{template}-{REMOVE_TEXT_AND_BACKGROUND}.png", "wb")
        file.write(res.content)
        file.close()
        return (id, True)
    return (id, False)


def createPool(ids):
    pool = mp.Pool(mp.cpu_count())
    pbar = tqdm(total=len(ids) * len(all_templates))

    all_results = pool.imap_unordered(getBitmoji, product(ids, all_templates))
    pool.close()

    generated = 0
    for res in all_results:
        current, found = res
        if (found): generated += 1
        pbar.set_description_str(f"Total: {generated} | Current : {current}")
        pbar.update(1)

    pool.join()
    return generated


def getExistingBitmojis(id):
    res = requests.get(f"https://render.bitstrips.com/render/7196007/{id}_2-v1.png")
    if res.status_code == 200 and not os.path.exists(f"./bitmoji/{id}"):
        return id
    return -1


def chunks(l, n):
    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n - 1):
        yield l[i * newn:i * newn + newn]
    yield l[n * newn - newn:]


if __name__ == '__main__':

    # Get all possibilities
    if not os.path.exists("./bitmoji"): os.mkdir("./bitmoji")
    with open("./all_templates.json") as file:
        all_templates = json.load(file)
        file.close()

    # Create X workers and each worker launch a pool
    print(f"#----- {WORKERS} Workers started  -----#")
    currs = futures.ThreadPoolExecutor(max_workers=WORKERS)
    currs_valid = futures.ThreadPoolExecutor(max_workers=WORKERS)

    # Get only existing bitmojis to gain time and save requests
    print(f" --- Validating {HOW_MUCH_BITMOJI} bitmojis ---")
    GLOBAL_LOOP = range(GLOBAL_START, GLOBAL_START - HOW_MUCH_BITMOJI, -1)
    GLOBAL_TO_CHECK = [e for e in currs_valid.map(getExistingBitmojis, GLOBAL_LOOP) if e > 0]
    GLOBAL_PER_WORKER = chunks(GLOBAL_TO_CHECK, WORKERS)

    print(f" --- Only {len(GLOBAL_TO_CHECK)} were valids ---")
    print(f"#----- Starting Pools -----#")

    # Generate all template for the validated ones
    wait_for = [
        currs.submit(createPool, next(GLOBAL_PER_WORKER))
        for _ in range(WORKERS)
    ]

    for f in futures.as_completed(wait_for): pass