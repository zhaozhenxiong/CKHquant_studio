"""Portable CLI over the same lightweight research services as the browser."""
import argparse
import json
from my_strategy.services.light_research import LightResearch

def main():
    parser=argparse.ArgumentParser(description='Kronos预测、Agent/MLP评分及持仓模拟')
    parser.add_argument('action',choices=['status','predict','score','simulate','daily','tune','observations'])
    parser.add_argument('--symbols',default='002436.SZ,600519.SH')
    parser.add_argument('--start',default='2026-01-01')
    parser.add_argument('--end')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    parser.add_argument('--samples',default=8,type=int,choices=[8,32,64])
    parser.add_argument('--manifest-id')
    parser.add_argument('--max-calls',type=int,default=200)
    parser.add_argument('--cost-stress',type=float,choices=[1.0,1.5,2.0],default=1.0)
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--model-type',choices=['mlp','l2'],default='mlp')
    parser.add_argument('--variant',choices=['A','B'],default='A')
    parser.add_argument('--score-model-id')
    parser.add_argument('--candidate-sessions',type=int,choices=[1,3],default=3)
    parser.add_argument('--fusion-mode',choices=['evidence','hard_filters'],default='evidence')
    parser.add_argument('--research-mode',choices=['historical_candidate','paper'],default='historical_candidate')
    parser.add_argument('--budget-seconds',type=int,default=300)
    parser.add_argument('--validation-dates',default='2025-09-01,2025-09-15')
    parser.add_argument('--test-dates',default='2025-10-09,2025-10-20')
    args=parser.parse_args();service=LightResearch()
    symbols=[s.strip() for s in args.symbols.split(',') if s.strip()]
    try:
        if args.action=='status':
            from my_strategy.services.kronos_prediction import asset_status
            result=asset_status()
        elif args.action=='predict':result=service.prediction(symbols[0],args.end,device=args.device,samples=args.samples)
        elif args.action=='tune':result=service.tune(symbols,args.validation_dates.split(','),args.test_dates.split(','),device=args.device)
        elif args.action=='observations':result=service.collect_score_observations(symbols,args.start,args.end,manifest_id=args.manifest_id,
            max_calls=args.max_calls,budget_seconds=min(args.budget_seconds,300),device=args.device)
        elif args.action=='score':
            manifest=service._read_checkpoint({'manifest_id':args.manifest_id},'score_observation_manifests') if args.manifest_id else None
            if args.manifest_id and not manifest:raise ValueError('冻结观察清单不存在')
            result=service.train_score(symbols,args.end or '2025-12-31',model_type=args.model_type,
                variant=args.variant,device=args.device,budget_seconds=min(args.budget_seconds,600),select=False,observation_manifest=manifest)
        elif args.action=='simulate':
            from my_strategy.adapters.czsc_adapter import latest_market_date
            result=service.backtest(symbols,args.start,args.end or latest_market_date(),device=args.device,samples=args.samples,save=args.save,
                score_model_id=args.score_model_id,score_model_type=args.model_type,candidate_sessions=args.candidate_sessions,
                fusion_mode=args.fusion_mode,research_mode=args.research_mode,budget_seconds=args.budget_seconds,cost_stress=args.cost_stress)
        else:
            from my_strategy.services.light_daily import sync_light_daily
            result=sync_light_daily(symbols=symbols,required_price_date=args.end,service=service)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    finally:service.close()

if __name__=='__main__':main()
