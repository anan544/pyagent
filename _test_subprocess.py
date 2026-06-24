import asyncio, locale

async def test():
    enc = locale.getpreferredencoding()  # cp936 on Chinese Windows
    # Use raw strings or proper escaping
    tests = [
        r'dir E:\thepython\super_study',
        r'dir "E:\thepython\super_study"',
        r'dir E:\thepython\super_study\super_register',
        r'echo hello',
        r'whoami',
    ]
    for cmd in tests:
        try:
            p = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(p.communicate(), timeout=5)
            out_s = out.decode(enc, errors='replace').strip() or '(empty)'
            err_s = err.decode(enc, errors='replace').strip() or '(empty)'
            print(f'rc={p.returncode} | {cmd}')
            print(f'  stdout: {out_s[:200]}')
            if err_s: print(f'  stderr: {err_s[:200]}')
        except Exception as e:
            print(f'ERR: {cmd} -> {e}')
        print()

asyncio.run(test())
