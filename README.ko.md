# 🧩 *"The Surprising Effectiveness of Deleting Weights in LLM Post-Training"* 확장 단상 — 개별 최고 성능의 삭제 마스크들을 신경망 직결(`C2C`)로 묶으면 '하나의 통합 AI'가 되지 않을까?

<p align="center">
  <strong>CKU Theory (Cache-to-Cache Mask Unification): 단일 공유 베이스 모델 위에서 순수 분야별 삭제 마스크들을 신경망 KV-Cache로 직결하는 통합 AI 가설</strong><br>
  <em>A Theoretical Hypothesis & Architectural Extension Note by <a href="https://axnli.com">AXNli Inc.</a> (Systems for Data & AI)</em>
</p>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-English-0ea5e9?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/언어-한국어_(Korean)-10b981?style=for-the-badge" alt="Korean"></a>
  <a href="https://arxiv.org/abs/2611.10866"><img src="https://img.shields.io/badge/Inspired_by-arXiv:2611.10866_(NeurIPS_2026)-b31b1b?style=for-the-badge" alt="arXiv 2611.10866"></a>
  <a href="https://axnli.com"><img src="https://img.shields.io/badge/Created_by-AXNli_Inc-6366f1?style=for-the-badge" alt="AXNli Inc."></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge" alt="MIT License"></a>
</p>

---

> 🌐 **For the primary English edition, please see [README.md](./README.md).**

---

## 1. 선행 연구에 대한 경의와 출발점

> [!NOTE]
> **선행 연구 출처 및 감사**  
> 본 아키텍처 이론 노트는 뉴욕대학교(NYU)와 캘거리대학교(University of Calgary) 연구진(**Jack Lu, Zhenbang Yang, Mike Lasby, Tejas Pote, Yani Ioannou, Mengye Ren**)이 NeurIPS 2026에 발표한 ***"The Surprising Effectiveness of Deleting Weights in LLM Post-Training"*** (`arXiv:2611.10866`, 공식 MIT 저장소: [`agentic-learning-ai-lab/deleting-weights`](https://github.com/agentic-learning-ai-lab/deleting-weights)) 및 칭화대학교 연구진의 **Cache-to-Cache (`C2C`)** 신경망 직접 소통 패러다임(`arXiv:2510.03215`)에서 깊은 영감을 받아 작성되었습니다. 훌륭한 발견과 깔끔한 오픈소스 코드를 공개해 주신 연구진께 깊은 경의를 표합니다.

Lu et al. (2026)은 사전학습된 언어모델에 새로운 가중치 숫자를 단 하나도 덧칠하지 않고, 오직 **`mlp.gate_proj` 가중치 중 `0.05%` 미만을 `0`으로 지우는 것(`Deletion Tuning`)만으로 전체 파인튜닝(`FFT`)에 필적하는 성능**을 달성할 수 있음을 보여주었습니다. 특히 두 가지 성질이 매우 인상적이었습니다.
1. **초경량 전문성**: 각 과목·분야의 전문성이 수 기가바이트의 모델 사본이 아니라 `packbits + zstd`로 압축된 **`~1 MB` 안팎의 초소형 비트맵 마스크** 하나에 담깁니다.
2. **원본 신경망 분포 보존**: 전체 가중치의 `99.95%` 이상이 원본 그대로 유지되며, 원본 모델과의 분포 이탈(`Forward KL ≤ 0.05`)이 극도로 작습니다.

---

## 2. 자연스럽게 떠오른 질문: "한 분야를 올렸을 때 다른 분야는, 그리고 만능 통합 AI는?"

논문의 과목별 `SFT` 실험(`MMLU-Pro` 14개 분야, `BBEH` 23개 과제별로 각각의 최적 마스크 학습)과 **6장(`Limitations and Future Work`)**을 읽으며 한 가지 자연스러운 질문이 떠올랐습니다.

> **수학 점수를 극대화하기 위해 삭제 튜닝을 적용해 수학 실력이 올랐다면, 수학이 아닌 다른 분야들과는 어떤 관계가 생길까? 나아가 수학·법률·코딩·과학을 모두 최고 수준으로 잘하는 '하나의 통합 AI'는 어떻게 만들 수 있을까?**

만약 서로 다른 여러 분야의 삭제 마스크들을 **단 하나의 고정된 마스크로 단순 합집합(Static Union)해 동시에 지워버리면**, 수학에서는 불필요했던 스위치가 법률이나 인문학에서는 중요한 신호를 담당하던 스위치일 수 있어 영역 간 미묘한 간섭(Trade-off)이 생길 수 있습니다.

여기서 발상을 전환해 다음과 같은 가설을 세워 보았습니다:

> **여러 분야를 하나의 고정 마스크로 억지로 뭉개어 합치지 말고, 각 분야별로 타협 없이 가장 뛰어난 삭제 마스크들을 개별 개발한 뒤, 같은 베이스 AI라 내부 신경망이 그대로 이어진다는 점을 이용해 `C2C (Cache-to-Cache)` 방식으로 연결하면 모든 것을 잘하는 '하나의 통합 AI'가 되지 않을까?**

---

## 3. 핵심 이론 가설: `CKU` (Cache-to-Cache Mask Unification)

원래 서로 다른 전문 AI 모델들을 **`C2C`(텍스트 재출력 없이 내부 `KV-Cache` 기억장을 직접 주고받는 방식)**로 묶으려면 **(1) 모델 개수만큼 메모리가 배수로 폭증하는 문제**와 **(2) 각 모델의 어텐션(`Q, K, V`) 좌표계가 달라져 별도의 번역 신경망(Projector)을 학습시켜야 하는 문제**가 있었습니다.

하지만 **`Deletion Tuning`으로 만든 전문가 마스크들은 모두 '단 하나의 똑같은 베이스 AI'에서 출발했기 때문에** 이 두 장벽이 동시에 사라집니다.

1. **어텐션 신경망(`Q, K, V, O`) 100% 동일 + `Forward KL ≤ 0.05`**:
   * 공개된 코드(`TARGET_MODULES = ["gate_proj"]`)에서 모든 어텐션 가중치는 1비트도 변하지 않고 고정됩니다. 따라서 법률 마스크가 문맥을 읽으며 쌓아 올린 `KV-Cache` 신경망 상태는 수학 마스크나 코딩 마스크 입장에서도 **이미 같은 신경망 언어(좌표계)**로 정렬되어 있습니다.
2. **모델 1개분 메모리로 수십 명의 최고 전문가가 한 몸처럼 작동**:
   * 메모리에 본체 AI는 **딱 1개**만 띄워 두고, 추론 단계에 따라 필요한 분야의 `~1 MB` 삭제 마스크 스위치만 마이크로초 단위로 전환하면서 **앞선 전문가가 형성한 `KV-Cache` 기억장을 그대로(`C2C`) 이어받아 추론을 완성**합니다.
   * 그 결과 타 분야 점수 하락이나 마스크 간 충돌 없이 각 분야의 최고 성능을 온전히 발휘하는 **'하나의 통합 AI(Unified AI)'**를 구현할 수 있습니다.

---

## 4. 인용 및 라이선스

* **창안 및 정리**: **[AXNli Inc. (`주식회사 에이엑스엔엘아이`)](https://axnli.com)**
* **연계 아키텍처**: **[CKN (`CKN-Architecture`)](https://github.com/kokogo100/CKN-Architecture)** · **[CKS (`CKS-Architecture`)](https://github.com/kokogo100/CKS-Architecture)** · **[CKD (`CKD-Architecture`)](https://github.com/kokogo100/CKD-Architecture)**
* **라이선스**: MIT License (© 2026 AXNli Inc.)
