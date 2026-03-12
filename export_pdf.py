"""
导出前四章为 PDF
用法：python3 export_pdf.py
输出：public/exports/datamining_ch01-04.pdf
"""
import asyncio
import subprocess
from pathlib import Path

BASE_URL = "http://localhost:3000"

# 按顺序列出所有需要导出的页面
PAGES = [
    ("/ch01",     "ch01_index"),
    ("/ch01/1.1", "ch01_1.1"),
    ("/ch01/1.2", "ch01_1.2"),
    ("/ch01/1.3", "ch01_1.3"),
    ("/ch02",     "ch02_index"),
    ("/ch02/2.1", "ch02_2.1"),
    ("/ch02/2.2", "ch02_2.2"),
    ("/ch02/2.3", "ch02_2.3"),
    ("/ch02/2.4", "ch02_2.4"),
    ("/ch02/2.5", "ch02_2.5"),
    ("/ch02/2.6", "ch02_2.6"),
    ("/ch02/2.7", "ch02_2.7"),
    ("/ch03",     "ch03_index"),
    ("/ch03/3.1", "ch03_3.1"),
    ("/ch03/3.2", "ch03_3.2"),
    ("/ch03/3.3", "ch03_3.3"),
    ("/ch03/3.4", "ch03_3.4"),
    ("/ch03/3.5", "ch03_3.5"),
    ("/ch03/3.6", "ch03_3.6"),
    ("/ch03/3.7", "ch03_3.7"),
    ("/ch04",     "ch04_index"),
    ("/ch04/4.1", "ch04_4.1"),
    ("/ch04/4.2", "ch04_4.2"),
    ("/ch04/4.3", "ch04_4.3"),
    ("/ch04/4.4", "ch04_4.4"),
    ("/ch04/4.5", "ch04_4.5"),
    ("/ch04/4.6", "ch04_4.6"),
    ("/ch04/4.7", "ch04_4.7"),
]

OUTPUT_DIR = Path("public/exports")
CHROMIUM = "/home/cican/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome"


async def export_pages():
    from playwright.async_api import async_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=CHROMIUM,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        page = await browser.new_page()

        for path, name in PAGES:
            url = BASE_URL + path
            print(f"  → {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # 等待 MathJax/KaTeX 渲染完成
            await page.wait_for_timeout(1500)

            out = OUTPUT_DIR / f"{name}.pdf"
            await page.pdf(
                path=str(out),
                format="A4",
                margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"},
                print_background=True,
            )
            pdf_files.append(str(out))
            print(f"     ✓ {out}")

        await browser.close()

    return pdf_files


def merge_pdfs(pdf_files):
    merged = "public/exports/datamining_ch01-04.pdf"
    try:
        import pypdf
        writer = pypdf.PdfWriter()
        for f in pdf_files:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                writer.add_page(page)
        with open(merged, "wb") as fh:
            writer.write(fh)
        print(f"\n✅ 合并完成：{merged}")
        return merged
    except ImportError:
        pass

    # fallback: pdfunite (poppler)
    result = subprocess.run(["which", "pdfunite"], capture_output=True)
    if result.returncode == 0:
        subprocess.run(["pdfunite"] + pdf_files + [merged], check=True)
        print(f"\n✅ 合并完成（pdfunite）：{merged}")
        return merged

    print(f"\n⚠️  未找到合并工具，各页 PDF 在 {OUTPUT_DIR}/")
    return None


if __name__ == "__main__":
    print("开始导出各页面 PDF...")
    files = asyncio.run(export_pages())
    print("\n正在合并...")
    merge_pdfs(files)
