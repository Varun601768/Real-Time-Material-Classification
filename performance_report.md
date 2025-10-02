# Performance Report — Real-Time Material Classification
- Project: End-to-End ML Pipeline for Scrap Material Classification
- Date: October 2, 2025
- Model: ResNet18 (Transfer Learning)
- Framework: PyTorch
## Key Metrics
- **Accuracy (overall):** 98.09%
- **Precision (macro avg):** 95.59%
- **Recall (macro avg):** 95.59%
- **F1-score (macro avg):** 95.59%
## Confusion Matrix
<img width="678" height="590" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c136ba01-8118-45e8-ad9f-3bbe31295bc1" />

- **Metal:** Precision 92.3%, Recall 92.3%
- **Plastic:** Precision 91.9, Recall 91.9%
- **E-waste:** Precision 99.2%, Recall 99.2%
- **Paper:** Precision 94.5%, Recall 94.5%
- **Fabric:** Precision 100%, Recall 100%

## Key Takeaways
- Model performs consistently across all 5 classes.
- Fabric has the highest recall (100%)
- Plastic has the lowest recall (91.9%) → dataset augmentation needed.
- Active learning should focus on collecting more **plastic** samples.
## Next Steps
- Add more data for plastic.
- Explore focal loss to handle class imbalance.
- Experiment with ResNet34 for deeper representation.
- Build other models to analyze the results.
