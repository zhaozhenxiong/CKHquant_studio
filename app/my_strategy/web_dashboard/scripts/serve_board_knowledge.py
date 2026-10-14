"""Run the board workspace without owning or interrupting model training."""
from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from my_strategy.web_dashboard.board_api import configure_board_routes
from my_strategy.web_dashboard.config import STATIC_ROOT


def create_board_app(*, db_path=None, raw_db_path=None):
    app = FastAPI(title='同花顺板块知识库')
    configure_board_routes(app, db_path=db_path, raw_db_path=raw_db_path)

    @app.middleware('http')
    async def security(request: Request, call_next):
        if request.url.path.startswith('/api/'):
            token = os.environ.get('KHQUANT_API_TOKEN', '')
            if token and not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
                return JSONResponse({'detail': '连接密钥无效'}, status_code=401)
            if request.method not in ('GET', 'HEAD', 'OPTIONS'):
                origin = request.headers.get('origin')
                if origin and urlsplit(origin).netloc != request.headers.get('host'):
                    return JSONResponse({'detail': '拒绝跨来源写请求'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'self'"
        return response

    @app.get('/')
    def home():
        return FileResponse(STATIC_ROOT / 'board_knowledge.html')

    app.mount('/static', StaticFiles(directory=STATIC_ROOT), name='static')
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='同花顺板块知识库')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8125)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--raw-db', type=Path)
    args = parser.parse_args(argv)
    if args.host not in {'127.0.0.1', 'localhost', '::1'} and not os.environ.get('KHQUANT_API_TOKEN'):
        raise ValueError('非本机监听须配置 KHQUANT_API_TOKEN')
    import uvicorn
    uvicorn.run(create_board_app(db_path=args.db, raw_db_path=args.raw_db), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
