Aqui está um `README.md` profissional, estruturado para impressionar recrutadores e entusiastas, incluindo a explicação técnica sobre a renderização visual em tempo real.

---

# 🌀 Jutsu Vision Pipeline: Classificador de Gestos Multiclasse

Um sistema de **visão computacional de baixa latência** que identifica gestos de mãos em tempo real utilizando Geometria Espacial, Aprendizado de Máquina (XGBoost) e Arquitetura Reativa para disparo de eventos multimídia.

---

## 🚀 Sobre o Projeto

O objetivo deste projeto foi criar uma interface homem-máquina (HMI) imersiva. Através da webcam, o sistema monitora seus movimentos, classifica gestos específicos (Jutsus) e dispara efeitos visuais e sonoros sincronizados.

Diferente de sistemas simples, este projeto lida com:

* **Detecção 3D** de pontos articulares.
* **Classificação Multiclasse** robusta.
* **Sincronia de eventos** (áudio e vídeo) sem travamentos.

---

## 🛠 Tecnologias

* **Visão Computacional:** `OpenCV`, `MediaPipe`.
* **Inteligência Artificial:** `XGBoost` (Gradient Boosting Trees).
* **Processamento de Dados:** `NumPy`, `Pandas`.
* **Áudio/Mídia:** `Pygame`.

---

## 🧠 Pipeline Técnico

### 1. Percepção e Geometria (MediaPipe)

Utilizamos o módulo `mediapipe.solutions.hands`. Ele processa o fluxo de vídeo e retorna um grafo de **21 pontos de referência** para cada mão.

* **Linhas e Bordas em Tempo Real:** O módulo `mp_drawing.draw_landmarks` desenha o esqueleto da mão (conexões entre pontos) e as bordas (landmarks) diretamente no frame. Isso é feito através de **interpolação geométrica**, onde o OpenCV desenha polígonos entre as coordenadas normalizadas retornadas pelo MediaPipe a cada frame (~30 FPS).

### 2. Engenharia de Features (Invariância à Posição)

Para tornar o sistema eficiente, transformamos coordenadas absolutas (pixels) em **coordenadas relativas ao pulso** (ponto 0).

* **Zero-Padding:** Implementamos um preenchimento matemático para garantir que o modelo aceite tanto gestos com uma mão quanto com duas, mantendo a consistência do vetor de entrada (126 dimensões).

### 3. O Classificador (XGBoost Multiclasse)

Treinado com `objective='multi:softprob'`, o modelo atua como um "cérebro" probabilístico, calculando a confiança de que o gesto atual pertence à classe Naruto, Shingeki ou Shikamaru.

### 4. Motor de Feedback (Máquina de Estados)

Para evitar que efeitos sonoros se sobreponham, desenvolvemos uma **Máquina de Estados** que:

* Detecta apenas a **borda de subida** (início do movimento).
* Aplica **Cooldown** (bloqueio temporal) para estabilidade.
* Renderiza animações via **Mesclagem Aditiva/Alpha**, permitindo que o fundo preto de assets de vídeo "desapareça" sobre a imagem da câmera.

---

## 📂 Estrutura do Repositório

```text
/
├── src/
│   ├── 1_coletar_dados.py    # Coleta coordenadas e exporta para CSV
│   ├── 2_treinar_modelo.py   # Treina o XGBoost e gera o .json
│   └── 3_main.py             # Pipeline de inferência e renderização
├── som/                      # Arquivos .wav/.mp3 de efeitos
├── frames_out/               # Frames para a animação do Kage Bunshin
├── images/                   # Assets (Brasões e logotipos)
└── dataset_jutsu.csv         # Dataset de treinamento

```

---

## ⚙️ Como executar

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/Jutsu-Vision-Pipeline.git
cd Jutsu-Vision-Pipeline

```


2. **Instale as dependências:**
```bash
pip install opencv-python mediapipe numpy xgboost pygame pandas

```


3. **Execute o sistema:**
```bash
python src/3_main.py

```



---

## 💡 Destaques para sua Apresentação

* **Engenharia de Dados:** Explique como o *Zero-Padding* resolveu a incompatibilidade de tensores entre poses de uma e duas mãos.
* **Arquitetura de Baixa Latência:** Mencione que o sistema de renderização utiliza `cv2.imshow` integrado com a lógica de *blending* de pixels, garantindo fluidez visual mesmo rodando inferência de IA.
* **Performance:** Destaque o uso do `XGBoost`, que permite rodar predições complexas em tempo real com uso mínimo de processamento de GPU.

---

*Desenvolvido por Daniel Duarte | Computer Engineering Student @ UFMA*