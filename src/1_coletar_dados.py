import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

ARQUIVO_CSV = 'dataset_jutsu.csv'

def extrair_features_relativas(multi_hand_landmarks):
    features = []
    sorted_hands = sorted(multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)
    for hand_landmarks in sorted_hands:
        pulso = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - pulso.x, lm.y - pulso.y, lm.z - pulso.z])
            
    # --- LÓGICA DE ZERO-PADDING NA INFERÊNCIA ---
    while len(features) < 126:
        features.append(0.0)
        
    return np.array(features)

cap = cv2.VideoCapture(0)
dados = []

print("=== MODO DE GRAVAÇÃO EM LOTE ===")
print("Pressione '1' -> Timer para pose do Jutsu (Alvo = 1)")
print("Pressione '2' -> Timer para pose SHINGEKI (Alvo = 2)")
print("Pressione '3' -> Timer para pose do SHIKAMARU (Alvo = 3)")
print("Pressione '0' -> Timer para poses aleatórias (Alvo = 0)")
print("Pressione 'q' -> Sair e salvar o CSV")

modo_gravacao = False
tempo_inicio = 0
classe_atual = -1
amostras_coletadas = 0
TOTAL_AMOSTRAS_POR_LOTE = 30 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1') and not modo_gravacao:
        modo_gravacao = True
        tempo_inicio = time.time()
        classe_atual = 1
        amostras_coletadas = 0
        print("\n[Timer Iniciado] Prepare-se para a pose do Jutsu do clone das sombras!")
        
    elif key == ord('2') and not modo_gravacao:
        modo_gravacao = True
        tempo_inicio = time.time()
        classe_atual = 2
        amostras_coletadas = 0
        print("\n[Timer Iniciado] Prepare-se para a pose do SHINGEKI!")
    
    elif key == ord('3') and not modo_gravacao:
        modo_gravacao = True
        tempo_inicio = time.time()
        classe_atual = 3
        amostras_coletadas = 0
        print("\n[Timer Iniciado] Prepare-se para a pose do Jutsu do clone do shikamaru!")

    elif key == ord('0') and not modo_gravacao:
        modo_gravacao = True
        tempo_inicio = time.time()
        classe_atual = 0
        amostras_coletadas = 0
        print("\n[Timer Iniciado] Prepare-se para as poses aleatórias!")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if modo_gravacao:
        tempo_decorrido = time.time() - tempo_inicio
        
        if tempo_decorrido < 3.0:
            segundos_restantes = 3 - int(tempo_decorrido)
            cv2.putText(frame, f"PREPARE-SE: {segundos_restantes}s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3, cv2.LINE_AA)
            
        elif amostras_coletadas < TOTAL_AMOSTRAS_POR_LOTE:
            cv2.putText(frame, f"GRAVANDO LOTE: {amostras_coletadas}/{TOTAL_AMOSTRAS_POR_LOTE}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 191, 0), 3, cv2.LINE_AA)
            
            # --- CORREÇÃO DO GARGALO AQUI: ACEITA 1 OU 2 MÃOS ---
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) in [1, 2]:
                features = extrair_features_relativas(results.multi_hand_landmarks).tolist()
                features.append(classe_atual)
                dados.append(features)
                amostras_coletadas += 1
                
        else:
            modo_gravacao = False
            print(f"Lote finalizado! O dataset agora possui {len(dados)} amostras totais.")

    cv2.imshow('Coletor de Dados - Multiclasse', frame)

cap.release()
cv2.destroyAllWindows()

if dados:
    colunas = [f'coord_{i}' for i in range(126)] + ['target'] 
    df = pd.DataFrame(dados, columns=colunas)
    df.to_csv(ARQUIVO_CSV, mode='a', header=not os.path.exists(ARQUIVO_CSV), index=False)
    print(f"\nSucesso! {len(dados)} novas amostras salvas no arquivo '{ARQUIVO_CSV}'.")
else:
    print("\nNenhuma amostra válida capturada. O CSV não foi alterado.")