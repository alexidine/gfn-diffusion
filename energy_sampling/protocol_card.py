"""A scannable HTML card of a config's training protocol: one panel per stage, a row per branch saying where its rows
come from, which loss terms it carries, which parameters it trains and its share, then the stage's rollout cadence,
draw, balance, learning rate, entry and exit. Generated from the parsed stages and the trainer's own loss-coefficient
overlay (config_snapshot._stage_summary / protocol.Stage), the same facts config_snapshot.describe_stage phrases as
prose, so the card cannot drift from what a run does.

    python -m config_snapshot configs/mk_dev.yaml --card protocol.html
"""
from __future__ import annotations

import copy
import html
from argparse import Namespace
from typing import Optional

import config_snapshot as cs

_DATA = {'fwd': 'own on-policy rollouts', 'replay': 'stored forward rollouts (replay buffer)'}
_BWD_DATA = {'dataset': 'prior dataset rows', 'prior': 'prior buffer rows'}
_CENTRES = {'mean': 'VarGrad: the group mean', 'head': 'TB against the detached Z(c) head',
            'head_stored': 'TB against ½ Z(c) head + ½ stored-row mean'}


def _n(v) -> str:
    return cs._num_text(v)


def _terms(c: dict) -> list[str]:
    """Nonzero loss-term weights. vg_by_condition only says how a VarGrad term groups its rows, so it is shown only
    beside a VarGrad term it modifies."""
    def on(k):
        v = c.get(k)
        return isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) != 0.0
    vg = on('vg_lb') or on('vg_lme')
    return [f'{k} {_n(c[k])}' for k in cs._LOSS_TERMS if on(k) and (k != 'vg_by_condition' or vg)]


def _trains(c: dict) -> list[str]:
    freeze_policy = float(c.get('freeze_policy', 0) or 0) > 0.5
    freeze_z = float(c.get('freeze_z', 0) or 0) > 0.5
    return [p for p, frozen in (('policy', freeze_policy), ('log Z', freeze_z)) if not frozen]


def _lr_text(spec: dict, args) -> str:
    sched = spec.get('lr_schedule')
    where = 'stage schedule'
    if sched is None:
        sched = cs._to_plain(getattr(getattr(args, 'lr_control', None), 'schedule', None))
        where = 'run schedule'
    seed = getattr(getattr(args, 'lr_control', None), 'seed_lr', None)
    if not sched:
        return f'{where}: none' + (f' (seed_lr {_n(seed)})' if seed is not None else '')
    segs, at = [], 0
    for s in sched:
        kind, to, steps = s.get('kind', 'geometric'), s.get('to'), int(s.get('steps', 0))
        segs.append(f"{'hold' if kind == 'hold' else kind} {_n(to)}× to step {at + steps}")
        at += steps
    return f'{where}: ' + ', then '.join(segs) + (f' (× seed_lr {_n(seed)})' if seed is not None else '')


def stage_card(spec: dict, index: int, args, *, successor: Optional[str] = None,
               n_stages: Optional[int] = None) -> dict:
    """The card's facts for one stage, from the same parse and overlay describe_stage uses."""
    import protocol as P

    spec = copy.deepcopy(spec)
    st = P.Stage(copy.deepcopy(spec), index)
    summ = cs._stage_summary(copy.deepcopy(spec), index, args)
    eff = summ['effective_loss_coeffs']
    note = None
    if isinstance(eff, str):
        note = f'base coefficients unavailable: {eff}'
        eff = {m: dict(st.loss_coeffs.get(m, {})) for m in cs.MODES}
    fc = eff.get('fwd') or {}
    pooled = isinstance(fc.get('pooled_vg'), (int, float)) and float(fc['pooled_vg']) > 0
    pooled_modes = {'bwd', fc.get('pooled_source', 'fwd')} if pooled else set()
    branches = []
    if st.train_mode == 'bwd':
        c = eff.get('bwd') or {}
        branches.append(dict(mode='bwd', rows=_BWD_DATA[st.bwd_sampling_mode], terms=_terms(c),
                             trains=_trains(c), share='all', inert=False, sidecar=False))
    else:
        total = float(sum(st.fracs.values())) if st.fracs else 0.0
        thr = st.deactivate_threshold
        if thr is None:
            thr = getattr(getattr(args, 'controller', None), 'deactivate_threshold', None)
            thr = 0.01 if thr is None else float(thr)
        for mode in cs.MODES:
            c = eff.get(mode) or {}
            rows = _DATA.get(mode) or _BWD_DATA[st.bwd_sampling_mode]
            terms = _terms(c)
            if mode in pooled_modes:
                terms = terms + ['pooled term']
            if mode == 'fwd' and st.fwd_z_sidecar:
                branches.append(dict(mode=mode, rows=rows, terms=terms, trains=_trains(c),
                                     share='sidecar ×1', inert=False, sidecar=True))
                continue
            share = float(st.fracs.get(mode, 0.0)) / total if total > 0 else None
            weighted = (share is None or share >= thr or float(st.min_fracs.get(mode, 0.0)) >= thr
                        or st.balance_can_raise(mode, thr))
            branches.append(dict(mode=mode, rows=rows, terms=terms if weighted else [],
                                 trains=_trains(c) if weighted else [],
                                 share=('carried' if share is None else f'{share:.2f}') if weighted else '0',
                                 inert=not weighted, sidecar=False))
    pooled_info = None
    if pooled:
        centre = fc.get('pooled_center', 'mean')
        pooled_info = dict(pairs=f"{fc.get('pooled_source', 'fwd')} rows with bwd rows",
                           centre=_CENTRES.get(centre, str(centre)),
                           detail=f"pooled_vg {_n(fc['pooled_vg'])}, bwd share {_n(fc.get('pooled_ratio', 0.5))}"
                                  + (f", beta {_n(fc['pooled_beta'])}" if 'pooled_beta' in fc else ''))
    mech = []
    if st.train_mode == 'fused':
        if st.fwd_rollout_every > 0:
            r = f'1 step in {st.fwd_rollout_every}'
            trig = {k: v for k, v in st.fwd_rollout_triggers.items() if v > 0}
            if trig:
                r += '; early on ' + ', '.join(f'{k} {_n(v)}' for k, v in trig.items())
            if st.replay_warmup_rows:
                r += f'; every step until replay holds {st.replay_warmup_rows:,} rows'
            if st.z_pin_rollout_every:
                r += f'; z-pin rollout every {st.z_pin_rollout_every}'
            mech.append(('rollouts', r, False))
        else:
            mech.append(('rollouts', 'every step', False))
    if st.condition_draw is not None:
        cd = st.condition_draw
        cn = cd['conditions'] or f"batch_size // {max(cd['replay_rows'], cd['prior_rows'])}"
        mech.append(('draw', f"{cn} conditions ({cd['pick']}), {cd['replay_rows']} replay + "
                             f"{cd['prior_rows']} bwd rows each", False))
    if st.balance is not None:
        btext, _ = cs._balance_text(spec.get('balance') or {}, st.balance)
        ann = st.balance.get('anneal_coeffs') or {}
        ec = getattr(args, 'energy_config', None)
        inert = bool(ann) and all(
            getattr(ec, k, None) is not None and float(getattr(ec, k)) >= float(v['target']) for k, v in ann.items())
        if inert:
            btext += ' — anneal inert: ' + ', '.join(f'{k} already {_n(getattr(ec, k))}' for k in ann)
        mech.append(('balance', btext, inert))
    for name, s in st.coeff_schedule.items():
        mech.append(('schedule', f"{name} → {_n(s['target'])} over {s['steps']} steps ({s['kind']})", False))
    mech.append(('learning rate', _lr_text(spec, args), False))
    actions = [f'{n}:{a}' if a else n for n, a in st.on_enter]
    entry = []
    if st.skip_if:
        entry.append(('skipped when', st.skip_if, False))
    if actions:
        entry.append(('on entry', ', '.join(actions) + (' (never fires: first stage)' if index == 0 else ''),
                      index == 0))
    exits = [f'{n}:{a}' if a else n for n, a in st.on_exit]
    if st.exit or st.max_steps is not None:
        when = []
        if st.exit:
            when.append(cs._exit_text(st.exit))
        if st.max_steps is not None:
            when.append(f'{st.max_steps:,} of its own steps')
        end = ' or '.join(when)
        then = [a for a in exits if a != 'stop']
        nxt = 'ends the run' if 'stop' in [n for n, _ in st.on_exit] else (
            f"→ {successor}" if successor else 'no successor')
        exit_row = dict(when=end, then=', '.join(then), next=nxt, terminal=False)
    else:
        exit_row = dict(when='none: runs until the epochs are spent', then='', next='', terminal=True)
    flags = [k for k, v in st.flags.items() if v]
    tail = [f'{k}={cs._verbatim(spec[k])}' for k in cs._TAIL_KEYS if spec.get(k) is not None and k != 'lr_schedule']
    return dict(name=st.name, index=index, n=n_stages, mode=st.train_mode, branches=branches, pooled=pooled_info,
                mech=mech, entry=entry, exit=exit_row, flags=flags, tail=tail, note=note)


def protocol_cards(args, name: Optional[str] = None) -> tuple[str, list[dict]]:
    """(protocol name, one stage_card per stage) for `name`, default the active protocol, on its own overlay."""
    from config_invariants import PROTOCOL_LIBRARY, PROTOCOL_SELECTOR
    name = name or getattr(args, PROTOCOL_SELECTOR)
    library = cs._to_plain(getattr(args, PROTOCOL_LIBRARY, None)) or {}
    specs = list((library.get(name) or {}).get('stages') or [])
    view = Namespace(**vars(args))
    setattr(view, PROTOCOL_SELECTOR, name)
    return name, [stage_card(s, i, view, successor=specs[i + 1].get('name') if i + 1 < len(specs) else None,
                             n_stages=len(specs)) for i, s in enumerate(specs)]


_CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--mute:#6b7280;--line:#e3e6ea;--head:#eef1f5;
--policy:#2563eb;--policy-bg:#e8efff;--z:#b45309;--z-bg:#fdf1e2;--data:#0f766e;--data-bg:#e3f5f2;
--term:#6d28d9;--term-bg:#f1eaff;--inert:#9ca3af;--accent:#334155;--exit:#be123c;--exit-bg:#fdecef}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1f25;--ink:#e6e9ee;--mute:#9aa3ad;--line:#2b323a;
--head:#212831;--policy:#7aa2ff;--policy-bg:#1d2a45;--z:#f0b46a;--z-bg:#3a2a16;--data:#5fd3c4;--data-bg:#143430;
--term:#c4a6ff;--term-bg:#2b2142;--inert:#5b636d;--accent:#cbd5e1;--exit:#ff7a93;--exit-bg:#3d1a22}}
*{box-sizing:border-box}body{margin:0;padding:28px 20px 48px;background:var(--bg);color:var(--ink);
font:15px/1.45 "Inter","Segoe UI",system-ui,-apple-system,sans-serif}
.wrap{max-width:1080px;margin:0 auto}h1{font-size:20px;margin:0 0 4px;font-weight:650}
.sub{color:var(--mute);margin-bottom:22px;font-size:13.5px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;
box-shadow:0 1px 2px rgba(0,0,0,.05)}
.shead{display:flex;align-items:baseline;gap:12px;padding:14px 18px;background:var(--head);border-bottom:1px solid var(--line)}
.idx{font-weight:700;color:var(--mute);font-size:13px;letter-spacing:.04em}
.sname{font-size:18px;font-weight:650}.mode{margin-left:auto;font-size:12px;font-weight:600;text-transform:uppercase;
letter-spacing:.06em;color:var(--accent);border:1px solid var(--line);border-radius:999px;padding:2px 10px}
.skipped{color:var(--z);font-size:13px;font-weight:600}
table{width:100%;border-collapse:collapse}th{text-align:left;font-size:11.5px;text-transform:uppercase;
letter-spacing:.06em;color:var(--mute);font-weight:600;padding:10px 18px 6px}
td{padding:9px 18px;border-top:1px solid var(--line);vertical-align:top}
td.b{font-weight:700;width:86px;font-family:"JetBrains Mono","Cascadia Code",Consolas,monospace;font-size:14px}
td.sh{width:120px;white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:600}
tr.inert td{color:var(--inert)}tr.inert td.b{text-decoration:line-through}
.chip{display:inline-block;border-radius:6px;padding:1px 8px;margin:1px 4px 1px 0;font-size:13px;font-weight:550}
.data{background:var(--data-bg);color:var(--data)}.term{background:var(--term-bg);color:var(--term)}
.pol{background:var(--policy-bg);color:var(--policy)}.lz{background:var(--z-bg);color:var(--z)}
.none{color:var(--inert);font-style:italic;font-size:13px}
.pooled{margin:0 18px 12px;padding:9px 12px;border-left:3px solid var(--term);background:var(--term-bg);
border-radius:6px;font-size:13.5px}
.grid{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;padding:12px 18px 14px;border-top:1px solid var(--line);
font-size:13.5px}.k{color:var(--mute);font-weight:600}.v.inert{color:var(--inert)}
.exitbar{display:flex;gap:10px;align-items:center;padding:11px 18px;border-top:1px solid var(--line);
background:var(--exit-bg);font-size:13.5px}.exitbar .k{color:var(--exit)}
.exitbar.term{background:var(--head)}
.arrow{text-align:center;color:var(--mute);font-size:22px;line-height:1;margin:8px 0}
.flags{padding:0 18px 12px}.flag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:999px;
padding:1px 9px;margin:2px 4px 2px 0;color:var(--accent)}
.tail{padding:0 18px 12px;color:var(--mute);font-size:12px;font-family:Consolas,monospace}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:-10px 0 18px;font-size:12.5px;color:var(--mute)}
"""


def _chips(items, cls) -> str:
    return ''.join(f'<span class="chip {cls}">{html.escape(t)}</span>' for t in items)


def render_html(args, name: Optional[str] = None, title: Optional[str] = None) -> str:
    pname, cards = protocol_cards(args, name)
    e = html.escape
    out = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
           f'content="width=device-width,initial-scale=1"><title>Protocol card</title><style>{_CSS}</style></head>'
           f'<body><div class="wrap"><h1>{e(title or "training protocol")}</h1>'
           f'<div class="sub">protocol <b>{e(pname)}</b> · {len(cards)} stage(s) · generated from the '
           f'config by protocol_card.py</div>'
           '<div class="legend"><span><span class="chip data">rows</span>where a branch\'s rows come from</span>'
           '<span><span class="chip term">term</span>loss terms</span><span><span class="chip pol">policy</span>'
           '<span class="chip lz">log Z</span>parameters it trains</span><span style="color:var(--inert)">'
           'grey = carries no weight / inert</span></div>']
    for i, c in enumerate(cards):
        if i:
            out.append('<div class="arrow">↓</div>')
        skip = next((v for k, v, _ in c['entry'] if k == 'skipped when'), None)
        out.append(f'<div class="stage"><div class="shead"><span class="idx">STAGE {c["index"] + 1}/{c["n"]}</span>'
                   f'<span class="sname">{e(c["name"])}</span>'
                   + (f'<span class="skipped">skipped when {e(skip)}</span>' if skip else '')
                   + f'<span class="mode">{e(c["mode"])}</span></div>')
        out.append('<table><tr><th>branch</th><th>rows from</th><th>loss</th><th>trains</th><th>share</th></tr>')
        for b in c['branches']:
            trains = ''.join(f'<span class="chip {"pol" if p == "policy" else "lz"}">{e(p)}</span>'
                             for p in b['trains']) or '<span class="none">nothing</span>'
            terms = _chips(b['terms'], 'term') or '<span class="none">no term</span>'
            out.append(f'<tr class="{"inert" if b["inert"] else ""}"><td class="b">{e(b["mode"])}</td>'
                       f'<td>{_chips([b["rows"]], "data")}</td><td>{terms}</td><td>{trains}</td>'
                       f'<td class="sh">{e(b["share"])}</td></tr>')
        out.append('</table>')
        if c['pooled']:
            p = c['pooled']
            out.append(f'<div class="pooled"><b>pooled term</b> pairs {e(p["pairs"])} · centre: '
                       f'<b>{e(p["centre"])}</b> · <span style="opacity:.8">{e(p["detail"])}</span></div>')
        rows = [(k, v, inert) for k, v, inert in c['mech']] + [(k, v, inert) for k, v, inert in c['entry']
                                                               if k != 'skipped when']
        out.append('<div class="grid">' + ''.join(
            f'<div class="k">{e(k)}</div><div class="v{" inert" if inert else ""}">{e(v)}</div>' for k, v, inert in rows)
            + '</div>')
        if c['flags']:
            out.append('<div class="flags">' + ''.join(f'<span class="flag">{e(f)}</span>' for f in c['flags'])
                       + '</div>')
        if c['tail']:
            out.append('<div class="tail">' + ' · '.join(e(t) for t in c['tail']) + '</div>')
        if c['note']:
            out.append(f'<div class="tail">{e(c["note"])}</div>')
        x = c['exit']
        out.append(f'<div class="exitbar{" term" if x["terminal"] else ""}"><span class="k">exit</span>'
                   f'<span>{e(x["when"])}</span>'
                   + (f'<span style="color:var(--mute)">then {e(x["then"])}</span>' if x['then'] else '')
                   + (f'<b style="margin-left:auto">{e(x["next"])}</b>' if x['next'] else '') + '</div></div>')
    out.append('</div></body></html>')
    return ''.join(out)
