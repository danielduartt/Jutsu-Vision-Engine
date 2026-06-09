import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import pygame
import time
import glob 

# 1. Configurações Iniciais de Visão e IA
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

try:
    model = xgb.XGBClassifier()
    model.load_model('modelo_kage_bunshin.json')
    print("[IA] Motor XGBoost Carregado.")
except:
    print("Erro: Arquivo 'modelo_kage_bunshin.json' não encontrado.")
    exit()

# 2. Configurações de Mídia (Áudio)
pygame.mixer.init()

# -- Áudio Naruto --
try:
    som_efeito = pygame.mixer.Sound('som/naruto_shadow_clones.mp3') 
    som_carregado = True
except:
    som_carregado = False
    print("[Aviso] Áudio 'som/naruto_shadow_clones.mp3' não encontrado.")

# -- Áudio Shingeki --
try:
    som_shingeki = pygame.mixer.Sound('som/shingeki.wav')
    som_shingeki_carregado = True
except:
    som_shingeki_carregado = False
    print("[Aviso] Áudio 'som/shingeki.wav' não encontrado.")

# 3. Configurações de Mídia (Visuais)

# -- Animação Naruto --
frames_animacao = []
try:
    caminhos_frames = sorted(glob.glob('frames_out/*.png'))
    if len(caminhos_frames) == 0:
        raise FileNotFoundError
        
    print(f"[Animação] Carregando {len(caminhos_frames)} quadros na VRAM...")
    for caminho in caminhos_frames:
        img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
        frames_animacao.append(img)
    animacao_carregada = True
    print(f"[Animação] {len(frames_animacao)} quadros carregados com sucesso!")
except:
    animacao_carregada = False
    print("[Aviso] Pasta 'frames_out' vazia. Rodando sem animação Naruto.")

# -- Imagem Shingeki --
try:
    img_shingeki = cv2.imread('images/brasao.png', cv2.IMREAD_UNCHANGED)
    if img_shingeki is not None:
        img_shingeki_carregada = True
        print("[Animação] Imagem do Shingeki carregada: images/brasao.png")
    else:
        raise FileNotFoundError
except:
    img_shingeki_carregada = False
    print("[Aviso] Imagem 'images/brasao.png' não encontrada.")

# -- Imagem Shikamaru --
try:
    img_shikamaru = cv2.imread('images/shikamaru.png', cv2.IMREAD_UNCHANGED)
    if img_shikamaru is not None:
        img_shikamaru_carregada = True
        print("[Animação] Imagem do Shikamaru carregada.")
    else:
        raise FileNotFoundError
except:
    img_shikamaru_carregada = False
    print("[Aviso] Imagem 'images/shikamaru.png' não encontrada.")

def extrair_features_relativas(multi_hand_landmarks):
    """Extrai features de 2 mãos (126 total). Se houver apenas 1, padding com zeros."""
    features = []
    sorted_hands = sorted(multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)
    
    for hand_idx in range(2):  # Sempre processa 2 mãos
        if hand_idx < len(sorted_hands):
            hand_landmarks = sorted_hands[hand_idx]
            pulso = hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                features.extend([lm.x - pulso.x, lm.y - pulso.y, lm.z - pulso.z])
        else:
            # Padding com zeros para mão faltante (21 pontos × 3 coordenadas)
            features.extend([0.0] * 63)
    
    return np.array(features)

def adicionar_imagem_transparente(fundo, overlay, x, y):
    bg_h, bg_w = fundo.shape[:2]
    
    if len(overlay.shape) == 3 and overlay.shape[2] == 4:
        h, w, c = overlay.shape
        tem_alfa = True
    else:
        h, w = overlay.shape[:2]
        tem_alfa = False

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return fundo

    x1, x2 = max(x, 0), min(x + w, bg_w)
    y1, y2 = max(y, 0), min(y + h, bg_h)
    
    overlay_x1, overlay_x2 = max(0, -x), min(w, bg_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, bg_h - y)

    overlay_recorte = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    fundo_recorte = fundo[y1:y2, x1:x2]

    if tem_alfa:
        alpha_s = overlay_recorte[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for color in range(0, 3):
            fundo[y1:y2, x1:x2, color] = (alpha_s * overlay_recorte[:, :, color] +
                                          alpha_l * fundo_recorte[:, :, color])
    else:
        fundo_aditivo = cv2.add(fundo_recorte, overlay_recorte)
        fundo[y1:y2, x1:x2] = fundo_aditivo

    return fundo

# 4. Pipeline de Tempo Real
print("[Sistema] Iniciando captura de vídeo...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

print("[Sistema] Câmera aberta. Pressione 'q' para sair.")

# Variáveis da Máquina de Estado atualizadas para strings
jutsu_ativo_antes = "NENHUM"
ultima_pose_ativada = "NENHUM"
tempo_ultimo_jutsu = 0
COOLDOWN = 1.5 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    status_text = "Procurando Padrao..."
    status_color = (0, 0, 255) 
    jutsu_agora = "NENHUM"

    if results.multi_hand_landmarks:
        # Desenha os landmarks (traços) das mãos
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # ACEITA 1 OU 2 MÃOS
        if len(results.multi_hand_landmarks) in [1, 2]:
            features = extrair_features_relativas(results.multi_hand_landmarks).reshape(1, -1)
            
            predicao = model.predict(features)[0]
            probabilidades = model.predict_proba(features)[0]

            # Proteção para garantir que o modelo tenha as 3 classes antes de acessar o índice
            prob_jutsu = probabilidades[1] if len(probabilidades) > 1 else 0
            prob_shingeki = probabilidades[2] if len(probabilidades) > 2 else 0
            prob_shikamaru = probabilidades[3] if len(probabilidades) > 3 else 0

            if predicao == 1 and prob_jutsu > 0.85:
                jutsu_agora = "NARUTO"
                status_text = f"KAGE BUNSHIN! ({prob_jutsu*100:.1f}%)"
                status_color = (255, 191, 0)

            elif predicao == 2 and prob_shingeki > 0.85:
                jutsu_agora = "SHINGEKI"
                status_text = f"SHINZOU WO SASAGEYO! ({prob_shingeki*100:.1f}%)"
                status_color = (0, 255, 0)

            elif predicao == 3 and prob_shikamaru > 0.85:
                jutsu_agora = "SHIKAMARU"
                status_text = f"SHIKAMARU! ({prob_shikamaru*100:.1f}%)"
                status_color = (200, 200, 50)
            

    # --- LÓGICA DE GATILHO (EDGE DETECTION) ---
    tempo_atual = time.time()
    
    if jutsu_agora != "NENHUM" and jutsu_agora != jutsu_ativo_antes and (tempo_atual - tempo_ultimo_jutsu > COOLDOWN):
        print(f"Ativando Efeitos Especiais: {jutsu_agora}")
        
        # Para qualquer áudio tocando para evitar sobreposição
        pygame.mixer.stop()
        
        if jutsu_agora == "NARUTO" and som_carregado:
            som_efeito.play()
        elif jutsu_agora == "SHINGEKI" and som_shingeki_carregado:
            som_shingeki.play()
            
        tempo_ultimo_jutsu = tempo_atual
        ultima_pose_ativada = jutsu_agora

    jutsu_ativo_antes = jutsu_agora

    # --- RENDERIZAÇÃO DAS ANIMAÇÕES E IMAGENS ---
    tempo_decorrido = tempo_atual - tempo_ultimo_jutsu
    
    if tempo_decorrido < COOLDOWN:
        
        # RENDERIZAÇÃO NARUTO (Animação Fluida)
        if ultima_pose_ativada == "NARUTO" and animacao_carregada:
            indice_atual = int((tempo_decorrido / COOLDOWN) * len(frames_animacao))
            if indice_atual < len(frames_animacao):
                frame_render = frames_animacao[indice_atual]
                
                h_tela, w_tela = frame.shape[:2]
                escala = (h_tela * 0.85) / frame_render.shape[0]
                novo_h = int(frame_render.shape[0] * escala)
                novo_w = int(frame_render.shape[1] * escala)
                frame_render = cv2.resize(frame_render, (novo_w, novo_h))
                
                x_centro = (w_tela - novo_w) // 2
                y_centro = (h_tela - novo_h) // 2
                frame = adicionar_imagem_transparente(frame, frame_render, x_centro, y_centro)

        # RENDERIZAÇÃO SHIKAMARU (Imagem Estática)
        elif ultima_pose_ativada == "SHIKAMARU" and img_shikamaru_carregada:
            h_tela, w_tela = frame.shape[:2]
            escala = (h_tela * 0.60) / img_shikamaru.shape[0]
            novo_h = int(img_shikamaru.shape[0] * escala)
            novo_w = int(img_shikamaru.shape[1] * escala)
            img_render = cv2.resize(img_shikamaru, (novo_w, novo_h))

            x_centro = (w_tela - novo_w) // 2
            y_centro = (h_tela - novo_h) // 2
            frame = adicionar_imagem_transparente(frame, img_render, x_centro, y_centro)

        # RENDERIZAÇÃO SHINGEKI (Imagem Estática)
        elif ultima_pose_ativada == "SHINGEKI" and img_shingeki_carregada:
            h_tela, w_tela = frame.shape[:2]
            escala = (h_tela * 0.60) / img_shingeki.shape[0]
            novo_h = int(img_shingeki.shape[0] * escala)
            novo_w = int(img_shingeki.shape[1] * escala)
            img_render = cv2.resize(img_shingeki, (novo_w, novo_h))

            x_centro = (w_tela - novo_w) // 2
            y_centro = (h_tela - novo_h) // 2
            frame = adicionar_imagem_transparente(frame, img_render, x_centro, y_centro)

        # Flash branco inicial para ambos os efeitos
        if tempo_decorrido < 0.1:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 10)

    # --- RENDERIZAÇÃO DO STATUS ---
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    cv2.imshow('Detector Multiclasse de Poses', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[Sistema] Encerrando...")
        break

cap.release()
cv2.destroyAllWindows()
print("[Sistema] Encerrado.")