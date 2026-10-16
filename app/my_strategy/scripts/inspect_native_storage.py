"""Write a read-only native object/reference inventory. This command never deletes."""
import argparse
import json
from pathlib import Path

from my_strategy.storage.native_inventory import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--database',type=Path)
    parser.add_argument('--alias-root',type=Path,action='append',default=[])
    parser.add_argument('--verify-content',action='store_true',help='Hash every native payload (potentially many GiB)')
    parser.add_argument('--max-reference-mib',type=int,default=512)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.max_reference_mib<1:parser.error('reference budget must be positive')
    result=inspect(args.project_root,database=args.database,alias_roots=args.alias_root,
        verify_content=args.verify_content,max_reference_bytes=args.max_reference_mib*1024**2)
    # Refuse any output target inside business data or artifact trees.
    output=args.output.expanduser().resolve();strategy=args.project_root.expanduser().resolve()/'my_strategy'
    if any(output.is_relative_to(strategy/name) for name in ('data','artifacts')):
        parser.error('inventory output must be outside business data and artifacts')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps({'output':str(output),'summary':result['summary'],'deletion_authorized':False},ensure_ascii=False))


if __name__=='__main__':main()
