import argparse
import cv2
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser(description='Extrai frames de um vídeo para uma pasta')
    p.add_argument('video', nargs='?', default='../videos/Smoke_Slow_motion_4k.mov', help='Caminho para o arquivo de vídeo')
    p.add_argument('-o', '--out', default='animacao_fumaca', help='Pasta de saída')
    p.add_argument('-p', '--prefix', default='frame', help='Prefixo dos arquivos de saída')
    p.add_argument('-s', '--step', type=int, default=1, help='Salvar a cada N frames (padrão 1)')
    p.add_argument('--start', type=int, default=0, help='Frame inicial (padrão 0)')
    p.add_argument('--end', type=int, default=None, help='Frame final (inclusivo)')
    p.add_argument('-f', '--format', default='png', choices=['png', 'jpg', 'jpeg'], help='Formato de saída')
    return p.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Erro: arquivo de vídeo não encontrado: {video_path}")
        sys.exit(1)

    print(f"Abrindo {video_path}...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo. Verifique o arquivo e codecs instalados.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pad = max(3, len(str(total_frames)))

    idx = 0
    saved = 0
    start = max(0, args.start)
    end = args.end if args.end is None else min(args.end, total_frames - 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < start:
            idx += 1
            continue

        if end is not None and idx > end:
            break

        if (idx - start) % args.step == 0:
            filename = f"{args.prefix}_{saved:0{pad}d}.{args.format}"
            out_path = out_dir / filename
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                print(f"Aviso: falha ao salvar {out_path}")
            else:
                if saved % 10 == 0:
                    print(f"Extraído: {out_path}")
                saved += 1

        idx += 1

    cap.release()
    print(f"Sucesso! {saved} frames extraídos na pasta '{out_dir}'.")


if __name__ == '__main__':
    main()