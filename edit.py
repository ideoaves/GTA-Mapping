import json, os, subprocess, sys, threading, webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000

ROOT = os.path.dirname(os.path.abspath(__file__))
busy = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=ROOT, **kw)

    def log_message(self, *args):
        pass

    # edit.html の微調整ボタンから届いた経路名で fine-tune.py を回し、ログを返す
    def do_POST(self):
        if self.path != '/editor/fine-tune':
            return self.send_error(404)
        names = [n for n in json.loads(self.rfile.read(int(self.headers['Content-Length']))) if n.replace('_', '').isalnum()]
        with busy:
            run = subprocess.run([sys.executable, '-u', 'editor/fine-tune.py', *names], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        log = '\n'.join(line[9:] for line in run.stdout.splitlines()) + ('\n' + run.stderr[-1500:] if run.returncode else '')
        print(log, flush=True)
        body = log.encode('utf-8')
        self.send_response(200 if run.returncode == 0 else 500)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# editor のページを開く。サーバーがまだ無ければこのプロセスで立て、wait なら閉じるまで動かし続ける
def open_page(page, wait):
    try:
        server = ThreadingHTTPServer(('localhost', PORT), Handler)
    except OSError:
        server = None
    url = f'http://localhost:{PORT}/editor/{page}'
    print(url, flush=True)
    webbrowser.open(url)
    if server and wait:
        server.serve_forever()
    elif server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == '__main__':
    open_page('edit.html', True)
