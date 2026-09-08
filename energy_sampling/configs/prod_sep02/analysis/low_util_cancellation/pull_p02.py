import wandb, json, pickle, os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
api = wandb.Api(timeout=300)
runs = list(api.runs("mkilgour/GFN Energy", filters={"config.tag": "p02"}, order="-created_at"))
for r in runs:
    fn = os.path.join(OUT, r.id + '.pkl')
    if os.path.exists(fn):
        print('cached', r.name); continue
    t0 = time.time()
    for attempt in range(4):
        try:
            rows = list(r.scan_history(page_size=5000))
            hist = pd.DataFrame(rows)
            ev = r.history(stream="events", samples=100000, pandas=True)
            break
        except Exception as e:
            print('retry', r.name, e); time.sleep(10)
    else:
        print('FAILED', r.name); continue
    meta = dict(id=r.id, name=r.name, state=r.state, created=r.created_at, host=(r.metadata or {}).get('host'),
                summary={k: v for k, v in dict(r.summary).items() if isinstance(v, (int, float, str))},
                config=dict(r.config), metadata=dict(r.metadata or {}))
    with open(fn, 'wb') as f:
        pickle.dump(dict(meta=meta, hist=hist, events=ev), f)
    print(r.name, r.state, 'hist', hist.shape, 'events', None if ev is None else ev.shape, '%.0fs' % (time.time() - t0), flush=True)
