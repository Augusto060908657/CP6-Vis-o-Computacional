# Guia do trabalho: YOLO treinado para o simulador de carro autônomo

## O que foi feito
1. **Correção no repositório:** as imagens `rainy-decoy.jpg` e `cat-cone.jpg` estavam com os nomes trocados em relação ao `src/main.tsx` (a de chuva tinha o nome do gato/cone e vice-versa). Os nomes foram trocados.
2. **Classes (7):** `red_light`, `green_light`, `pedestrian`, `crosswalk`, `dog`, `cat`, `traffic_cone`. Os nomes são reconhecidos por `category_for()` em `server.py`, sem alterar o servidor.
3. **Anotação:** caixas marcadas manualmente nas 5 cenas (`training/annotations.py`). A esfera verde da vitrine na cena de chuva NÃO foi marcada como semáforo (é o "decoy").
4. **Dataset:** `training/build_dataset.py` gera 75 imagens de treino (5 originais + 70 variações com recorte, brilho, contraste, saturação e espelhamento; sem alterar matiz, para não trocar vermelho por verde) e 5 de validação.
5. **Treino:** `training/train.py`, fine-tuning do `yolov8n.pt` (COCO), 40 épocas, imgsz 960, AdamW, `hsv_h=0`.
6. **Modelo final:** `models/autonomous-car-yolov8.pt`. Ele vem antes do `yolov8n.pt` em ordem alfabética, então o servidor o carrega por padrão.

## Resultados medidos (código real do server.py, conf 0.25)
| Cena | Esperado | Baseline yolov8n | Modelo treinado |
|---|---|---|---|
| clear-green | GO | GO (100) | GO (100) |
| red-pedestrian | STOP | STOP (87, sem crosswalk) | STOP (100) |
| dog-crossing | STOP | STOP (100) | STOP (100) |
| rainy-decoy | STOP | STOP (100) | STOP (100) |
| cat-cone | STOP | STOP (60, não achou gato nem cone) | STOP (100) |
| **Média** | | **89,4** | **100,0** |

Validação do treino (mAP50 = 0,99; mAP50-95 = 0,905). Gráficos em `docs/` (results.png, confusion_matrix.png, val_batch*_pred.jpg).

## LIMITAÇÃO (declare no relatório)
Só existem 5 imagens e o jogo avalia exatamente nelas. A validação usa as mesmas cenas do treino, portanto os números mostram que o modelo aprendeu essas cenas, não que generaliza para imagens novas. Para generalizar seria necessário um dataset maior e separado (treino/validação/teste).

## Como rodar
```bash
uv sync                      # dependências Python (ou: pip install ultralytics pillow)
npm install
uv run python server.py      # terminal 1
npm run dev                  # terminal 2 -> http://localhost:5173
```
No jogo, confira que o modelo selecionado é `autonomous-car-yolov8.pt` e rode as dificuldades Easy, Medium e Hard.

## Prints para entregar
- Easy: cena clear-green (GO) e red-pedestrian (STOP)
- Medium: dog-crossing e cat-cone
- Hard: rainy-decoy
- Relatório final (RUN COMPLETE) de cada dificuldade, com a pontuação
Opcional: print do baseline (selecione `yolov8n.pt`) para comparar.

## Para retreinar
```bash
python training/build_dataset.py
python training/train.py --epochs 40 --imgsz 960 --device cpu   # ou --device 0 com GPU
```
