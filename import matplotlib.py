import matplotlib.pyplot as plt
import numpy as np

# --- 여기에 사용자 Loss 데이터 입력 ---
# 예시: Loss 값을 담은 리스트 (실제로는 훈련 과정에서 얻은 값을 사용하세요)
loss_data = [
    0.85, 0.72, 0.65, 0.59, 0.51, 0.45, 0.38, 0.32, 0.28, 0.25,
    0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11
]
# 에폭 또는 반복 횟수 (Loss 데이터의 길이와 일치해야 합니다)
epochs = range(1, len(loss_data) + 1)
# -----------------------------------

# 1. 그래프 생성
plt.figure(figsize=(10, 6)) # 그래프 크기 설정
plt.plot(epochs, loss_data, label='Training Loss', color='blue', marker='o', linestyle='-')

# 2. 그래프 제목 및 축 레이블 설정
plt.title('Quantum AI Model Loss Curve', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)

# 3. 그리드 추가 (선택 사항)
plt.grid(True, linestyle='--', alpha=0.7)

# 4. **PNG 파일로 저장** (가장 중요한 부분!)
# 저장할 파일명과 경로를 지정합니다. 확장자를 .png로 지정합니다.
# `dpi`는 해상도(dots per inch)를 설정하며, 높을수록 화질이 좋습니다.
# `bbox_inches='tight'`는 그래프 주변의 불필요한 공백을 제거해줍니다.
save_filename = 'quantum_ai_loss_curve.png'
plt.savefig(save_filename, dpi=300, bbox_inches='tight')

print(f"✅ Loss Curve가 '{save_filename}'으로 저장되었습니다.")

# (선택 사항) 저장 후 그래프를 화면에 표시
# plt.show()