from pathlib import Path
import sys

src = Path(sys.argv[1]).read_text(encoding="utf-8")
Path("E:/Proj_site/app.py").write_text(src, encoding="utf-8")
print("Wrote E:/Proj_site/app.py")
