"""Observe existing jobs only. Never launch, terminate, restart or change a model."""
import argparse,json,time,datetime
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None
def capture(version,expected):
    base=ROOT/'outputs'/version;pipeline=read(base/'pipeline_status.json');training=read(base/'taoca/status.json');analysis=read(base/'analysis_status.json');result=dict(pipeline=pipeline,training=training,analysis=analysis,live_handles=[])
    if pipeline and pipeline.get('pid'):
        try:
            manager=psutil.Process(pipeline['pid']);cmd=manager.cmdline()
            if manager.is_running() and any(token.endswith(expected) for token in cmd):
                for p in [manager]+manager.children(recursive=True):
                    try:
                        if p.is_running() and p.name().lower() in ['python.exe','pythonw.exe','python','python3']:
                            result['live_handles'].append(dict(pid=p.pid,command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),rss_bytes=p.memory_info().rss))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            else:result['manager_identity_mismatch_or_not_live']=True
        except psutil.NoSuchProcess:result['manager_handle_missing']=True
        except psutil.AccessDenied:result['process_observation_unavailable']=True
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=120);p.add_argument('--interval',type=float,default=45);a=p.parse_args();deadline=time.monotonic()+a.minutes*60
    while True:
        versions={v:capture(v,e) for v,e in [('physical_v7',('run_physical_v7_local.py','resume_physical_v7_checked.py')),('metric_v8',('run_metric_v8_local.py',))]}
        print(json.dumps(dict(observed_at=datetime.datetime.now().isoformat(),versions=versions)),flush=True)
        states=[(r['pipeline'] or {}).get('state') for r in versions.values()]
        if 'failed' in states:print(json.dumps(dict(monitor_finished='pipeline_failure_requires_diagnosis')),flush=True);return
        if states[0]=='complete' and states[1] in ['complete','not_needed_v7_met_accuracy_criterion']:
            print(json.dumps(dict(monitor_finished='pipelines_terminal_accuracy_still_requires_audit')),flush=True);return
        if time.monotonic()>=deadline:print(json.dumps(dict(monitor_finished='observation_deadline_not_a_terminal_training_state')),flush=True);return
        time.sleep(min(a.interval,deadline-time.monotonic()))
if __name__=='__main__':main()
