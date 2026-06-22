# Propozycje Usprawnień Systemu CARIS

## 1. Fine-tuning modeli (QLoRA)
* **Cel**: Dostosowanie modeli detekcji (YOLO) i VLM (Gemma) do specyfiki obrazów malowanych/rycin.
* **Metoda**: Trening adapterów LoRA w precyzji 4-bitowej (QLoRA).
* **Ograniczenia**: Wymagana pamięć VRAM GPU (min. 16 GB dla Gemma-12B - prawdopodobnie konieczne będzie skorzystanie z chmury, np. Google Colab).

## 2. Filtrowanie odległości wektorowej (ChromaDB)
* **Cel**: Eliminacja fałszywych dopasowań semantycznych (False Positives).
* **Metoda**: Odczyt wartości `"distances"` zwracanych przez zapytanie ChromaDB. Wprowadzenie progu odcięcia (np. `distance < 0.8` dla metryki L2 lub cosinusowej) w funkcji `get_iconclass_codes_embeddings`.
* **Implementacja**: Dodanie parametru konfiguracyjnego `DISTANCE_THRESHOLD` w `config.py`.

## 3. Hierarchiczne osadzanie wektorowe (Contextual Embeddings)
* **Cel**: Rozwiązanie problemu wieloznaczności pojęć w bazie wektorowej poprzez dodanie kontekstu nadrzędnego.
* **Metoda**: Zmiana formatu wejściowego dokumentu w procesie generowania bazy wektorowej. Zamiast osadzać samą definicję (np. `apple`), osadzana będzie pełna ścieżka klasyfikacji (np. `Nature -> plants -> fruits -> apple`).
* **Implementacja**: Rekurencyjna rekonstrukcja ścieżki rodziców dla każdego kodu w `embeddings.py`.

## 4. Weryfikacja końcowa kandydatów (Reranking)
* **Cel**: Ostateczna filtracja wygenerowanych kodów Iconclass na podstawie całego kontekstu obrazu.
* **Metoda**:
  * **Wariant A (LLM/VLM)**: Przekazanie listy wygenerowanych kodów i opisu obrazu do modelu Gemma z promptem klasyfikującym (Zero-shot binary classification).
  * **Wariant B (Cross-Encoder)**: Zastosowanie lekkiego modelu Cross-Encoder do ponownego przeliczenia korelacji między opisem a kodami.

## 5. Abstrakcja pojęć i symbolika
* **Cel**: Mapowanie zestawu podstawowych obiektów na pojęcia wyższego rzędu (np. `psy + konie + broń` -> `polowanie`).
* **Metoda**:
  * **Wariant A (LLM)**: Promptowanie LLM o syntezę i ekstrakcję motywu przewodniego na podstawie listy wykrytych obiektów.
  * **Wariant B (Średnia wektorów)**: Wyznaczenie centroidu (średniej geometrycznej) wektorów osadzeń wykrytych tagów i odpytanie bazy ChromaDB wektorem średnim.
  * **Wariant C (Reguły asocjacyjne)**: Zastosowanie algorytmu *Apriori* na zbiorze `tune_data` w celu automatycznego wyznaczenia reguł powiązań tagów z kodami Iconclass.
