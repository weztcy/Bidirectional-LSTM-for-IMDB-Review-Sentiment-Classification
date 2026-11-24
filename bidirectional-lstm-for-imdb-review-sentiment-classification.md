# Portfolio Project Data — Bidirectional LSTM for IMDB Review Sentiment Classification

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/bidirectional_lstm_for_imdb/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image07.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "bidirectional-lstm-for-imdb-review-sentiment-classification",
  "code": "PRO-37 // ML-22",
  "title": "Bidirectional LSTM for IMDB Review Sentiment Classification",
  "category": "machine-learning",
  "categoryLabel": "Deep Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Bidirectional-LSTM-for-IMDB-Review-Sentiment-Classification",
  "license": "MIT License",
  "banner": "/images/projects/bidirectional_lstm_for_imdb/banner.webp",
  "summary": "PyTorch Bidirectional LSTM for binary IMDb review sentiment classification, benchmarked against TF-IDF Logistic Regression with Streamlit inference.",
  "projectDescription": "Built a PyTorch Bidirectional LSTM with global max pooling to classify 50,000 IMDb movie reviews into positive or negative sentiment. Benchmarked against TF-IDF Logistic Regression on an untouched 25,000-review test split, achieving 87.84% accuracy and 0.9494 ROC-AUC with Streamlit inference.",
  "architectureSummary": "HTML & Negation Normalization -> 30k Train-Only Vocab (256 Tokens) -> 128d Embedding -> BiLSTM (128x2) -> Global Max Pool -> Dense ReLU Classifier -> Sigmoid Output",
  "techStack": [
    "Python",
    "PyTorch",
    "Scikit-Learn",
    "NumPy",
    "Pandas",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Joblib"
  ],
  "telemetryMatrix": {
    "projectType": "Deep Sequence NLP Classifier",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Natural Language Processing",
    "architecturePipeline": "Negation Prep -> BiLSTM -> MaxPool -> App",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Deep Sequence Classification Pipeline",
    "learningParadigm": "Deep Learning",
    "modelDomain": "Movie Review Sentiment Analysis",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Bidirectional Sequence Modeling",
      "Negation-Preserving Tokenization",
      "Streamlit Live Inference"
    ]
  },
  "mlProblemContext": "Movie review sentiment classification demands tracking long-range discourse, syntactic inversions, and negation scopes across unstructured text. Standard bag-of-words baselines ignore token order, so a Bidirectional LSTM is formulated to read sequences forward and backward without test-set or vocabulary leakage.",
  "mlDatasetOverview": {
    "profileBadge": "IMDb v1.0 50k Corpus",
    "source": "Stanford Large Movie Review Dataset (Maas et al., 2011) under dataset/",
    "modality": "NLP",
    "scale": "50,000 Labeled Reviews (20k Train / 5k Val / 25k Test)",
    "inputShape": "N x 256 Integer Token IDs",
    "dataFormat": "Raw .txt Review Files & Pickle Split Cache",
    "description": "Balanced binary corpus of 25,000 positive and 25,000 negative movie reviews. Stratified 80/20 splitting (seed 42) forms 20,000 training and 5,000 validation reviews while keeping all 25,000 test reviews isolated and excluding the 50,000 unlabeled reviews.",
    "provenanceNote": "Public academic benchmark by Maas et al. (ACL 2011); programmatically verified for UTF-8 decoding with zero empty files."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Token Sequence & Polarity Space",
    "spaceNote": "30,000 train-only vocabulary indices truncated or padded to 256 tokens",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Review Sentiment Polarity",
      "type": "Binary Class",
      "distribution": "Exact 50.0% Negative (0) / 50.0% Positive (1) across train, val, and test splits"
    },
    "inputSignals": [
      {
        "name": "Word & Punctuation Token IDs",
        "type": "Discrete Sequence Tensor",
        "importanceOrRole": "Primary sequential input (256 steps)",
        "description": "Integer indices from a 30,000-token vocabulary preserving expressive punctuation (!, ?) alongside words."
      },
      {
        "name": "Expanded Negation Tokens",
        "type": "Normalized Lexical Cue",
        "importanceOrRole": "Critical polarity inverter",
        "description": "Contracted forms such as can't, won't, and n't expanded into explicit negation tokens prior to indexing."
      },
      {
        "name": "Forward & Backward Hidden States",
        "type": "Bidirectional Context Vector",
        "importanceOrRole": "256-dim temporal representation",
        "description": "Concatenated left-to-right and right-to-left LSTM states capturing preceding and succeeding clause context."
      },
      {
        "name": "Global Max-Pooled Features",
        "type": "Aggregated Sequence Vector",
        "importanceOrRole": "Salient affective signal selector",
        "description": "Element-wise maximum over the 256 time steps isolating the strongest sentiment activations across the review."
      }
    ],
    "dataCharacteristicsSummary": "Training reviews are strongly right-skewed, spanning 10 to 2,470 words with a 174-word median and 233.8-word mean. Setting the sequence limit to 256 tokens covers roughly 71% of reviews without truncation while retaining high-frequency negation markers such as not and but in the 30,000-token vocabulary."
  },
  "mlLearningApproach": {
    "paradigm": "Deep Sequence Classification via Bidirectional Recurrent Neural Networks",
    "methodologyDescription": "The primary model trains a 4.12M-parameter PyTorch BiLSTMClassifier end-to-end from scratch with spatial dropout and global max pooling. To establish a rigorous empirical reference, a sublinear TF-IDF unigram-bigram Logistic Regression baseline is trained on identical splits.",
    "justification": "Bidirectional recurrence captures left-and-right syntactic negation contexts that bag-of-words models miss, while global max pooling surfaces decisive sentiment clauses anywhere within the 256-token window without requiring attention overhead."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Stratified Pool Partitioning",
      "description": "Split the 25,000 labeled training reviews into 20,000 train and 5,000 validation records using stratified sampling (seed 42), keeping the 25,000 test reviews untouched.",
      "badge": "Split Isolation"
    },
    {
      "step": "02",
      "title": "HTML Decoding & Markup Stripping",
      "description": "Unescape HTML entities (&amp;, &quot;) and replace <br /> line breaks and residual tags with single whitespace characters.",
      "badge": "Text Cleaning"
    },
    {
      "step": "03",
      "title": "Contraction & Negation Expansion",
      "description": "Expand English contractions (can't -> can not, won't -> will not, n't -> not) and lowercase text while preserving sentiment punctuation.",
      "badge": "Negation Cue"
    },
    {
      "step": "04",
      "title": "Train-Only Vocabulary Indexing",
      "description": "Tokenize via regex and build a 30,000-token vocabulary (min_freq=2, <pad>=0, <unk>=1) exclusively from the 20,000 training reviews.",
      "badge": "Leakage Guard"
    },
    {
      "step": "05",
      "title": "Sequence Truncation & Padding",
      "description": "Convert token lists to integer sequences and apply post-truncation and post-padding to uniform length T = 256.",
      "badge": "Tensor Shaping"
    }
  ],
  "mlDataPreparationNote": "Both the BiLSTM vocabulary (models/vocab.json) and the baseline TF-IDF vectorizer are fitted strictly on the 20,000 training reviews; validation and test splits are only transformed at inference time.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Bidirectional LSTM Classifier (BiLSTMClassifier)",
    "architectureSummary": "128-dim Embedding (30,000 vocab, padding_idx=0) -> Dropout (0.30) -> 1-layer Bidirectional LSTM (128 hidden/direction, 256 concat) -> Global Max Pooling -> Dense Linear (256->64) + ReLU + Dropout (0.30) -> Linear (64->1)",
    "configurationBadge": "4.12M Params",
    "hyperparameters": [
      {
        "name": "Embedding & Hidden Dim",
        "value": "128 / 128 (x2)",
        "description": "128-dimensional learned embeddings feeding 128 forward and 128 backward LSTM units."
      },
      {
        "name": "Max Sequence Length",
        "value": "256 tokens",
        "description": "Uniform post-padded and post-truncated sequence length balancing context and CPU speed."
      },
      {
        "name": "Vocabulary Size",
        "value": "30,000 (min_freq=2)",
        "description": "Most frequent training tokens plus reserved <pad> (0) and <unk> (1) indices."
      },
      {
        "name": "Dropout Regularization",
        "value": "0.30",
        "description": "Applied after the embedding layer and dense hidden layer to curb overfitting."
      },
      {
        "name": "Batch Size & Gradient Clip",
        "value": "64 / Max Norm 5.0",
        "description": "Mini-batch size and L2 gradient norm ceiling preventing exploding LSTM gradients."
      },
      {
        "name": "LR & Weight Decay",
        "value": "1e-3 / 1e-5",
        "description": "Initial Adam learning rate with L2 weight decay and ReduceLROnPlateau (factor 0.5)."
      }
    ],
    "objectiveCriterion": "Binary Cross-Entropy with Logits (BCEWithLogitsLoss)",
    "optimizationSolver": "Adam Optimizer with ReduceLROnPlateau & Gradient Clipping",
    "trainingProcess": "Trained for 8 epochs (751.7s on CPU) with seed 42, early stopping patience 3, and validation loss checkpointing. Validation loss reached its minimum of 0.3167 (87.34% val accuracy) at Epoch 5 before overfitting triggered LR halving and early stopping at Epoch 8."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Stratified Validation Holdout + Untouched 25k Test Benchmark",
    "description": "Epoch selection and early stopping rely exclusively on the 5,000-review stratified validation split. Final unbiased metrics are computed once on the untouched 25,000-review test set and directly compared against the TF-IDF + Logistic Regression baseline.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "87.84%",
        "benchmark": "Baseline: 89.78%"
      },
      {
        "label": "TEST PRECISION",
        "value": "87.62%",
        "benchmark": "Baseline: 89.39%"
      },
      {
        "label": "TEST RECALL",
        "value": "88.14%",
        "benchmark": "Baseline: 90.28%"
      },
      {
        "label": "TEST F1-SCORE",
        "value": "87.88%",
        "benchmark": "Baseline: 89.83%"
      },
      {
        "label": "TEST ROC-AUC",
        "value": "0.9494",
        "benchmark": "Baseline: 0.9612"
      },
      {
        "label": "TEST BCE LOSS",
        "value": "0.3003",
        "benchmark": "Target < 0.3500"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Live Sentiment Prediction Interface",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image01.webp",
      "telemetryNote": "Real-time review classification showing 99.67% positive confidence and normalized token count."
    },
    {
      "title": "Holdout Evaluation & Baseline Comparison View",
      "badge": "Metrics Dashboard",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image02.webp",
      "telemetryNote": "Displays 87.84% accuracy, 0.9494 ROC-AUC, confusion matrix counts, and TF-IDF comparison."
    },
    {
      "title": "Methodology & Deep Architecture Tab",
      "badge": "Pipeline Docs",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image03.webp",
      "telemetryNote": "Documents contraction expansion, 128d embedding, 256d BiLSTM, and Adam regularization."
    },
    {
      "title": "Training Curves & Top Corpus Token Frequencies",
      "badge": "Convergence & EDA",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image04.webp",
      "telemetryNote": "Best checkpoint marked at Epoch 5 (0.3167 val loss) alongside top 25 frequent corpus tokens."
    },
    {
      "title": "Review Length Histogram & Test Confusion Matrix",
      "badge": "Distribution & CM",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image05.webp",
      "telemetryNote": "256-token cap covers 70.7% of reviews; test confusion matrix records 10,944 TN and 11,017 TP."
    },
    {
      "title": "Test Set ROC & Precision-Recall Curves",
      "badge": "Threshold Curves",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image06.webp",
      "telemetryNote": "BiLSTM achieves 0.9494 ROC-AUC and 0.9479 AP versus 0.9612 ROC-AUC and 0.9599 AP baseline."
    },
    {
      "title": "Multi-Metric Model Comparison & Split Balance",
      "badge": "Benchmark & Splits",
      "imageUrl": "/images/projects/bidirectional_lstm_for_imdb/image07.webp",
      "telemetryNote": "Contrasts all five test metrics and verifies exact 50/50 class balance across all three splits."
    }
  ],
  "mlVisualizationTelemetryNote": "Training and validation loss diverge smoothly after Epoch 5, confirming that early stopping preserves peak generalization (87.84% test accuracy vs. 87.34% val accuracy).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Error analysis across all 25,000 test predictions isolated 1,556 false positives and 1,483 false negatives. While bidirectional states handle local negation well, global max pooling over 256 tokens lets dense pockets of superlative or grim tokens dominate ironic or mixed-sentiment reviews.",
    "keyDrivers": [
      {
        "factor": "Expanded Negation Preservation",
        "impact": "Retaining explicit negation tokens enables accurate polarity flips in multi-clause reviews."
      },
      {
        "factor": "Global Max Pooling Aggregation",
        "impact": "Captures strong local praise or criticism anywhere in the sequence, but can over-weight early praise in two-part reviews."
      },
      {
        "factor": "256-Token Sequence Truncation",
        "impact": "Covers ~71% of reviews cleanly on CPU while discarding late verdict paragraphs in longer documents."
      },
      {
        "factor": "Train-From-Scratch Embeddings",
        "impact": "Fits 30k domain tokens on 20k reviews without external priors, trailing full-document TF-IDF n-grams by 1.94% accuracy."
      }
    ],
    "limitations": [
      {
        "title": "Sarcasm & Ironic Praise Vulnerability",
        "description": "Satirical negative reviews packed with positive adjectives (e.g., 'masterpiece', '10/10') trigger high-confidence false positives without world knowledge."
      },
      {
        "title": "Genre-Specific Negative Lexicon",
        "description": "Positive horror or B-movie reviews praising 'gruesome' or 'badly acted' camp qualities frequently produce false negatives."
      },
      {
        "title": "Long-Document Tail Truncation",
        "description": "Reviews longer than 256 tokens lose trailing conclusion sentences, whereas the TF-IDF baseline scores the entire text."
      }
    ],
    "futureImprovements": [
      {
        "title": "Pretrained Subword or Static Embeddings",
        "target": "Initialize embeddings with GloVe 300d or FastText to boost out-of-sample semantic coverage past 90% accuracy."
      },
      {
        "title": "Additive Self-Attention Pooling",
        "target": "Replace global max pooling with interpretable token-level attention weights to balance two-part reviews."
      },
      {
        "title": "Transformer Encoder Benchmarking",
        "target": "Fine-tune DistilBERT or RoBERTa to resolve pragmatic sarcasm and longer 512-token review contexts."
      }
    ],
    "discoveryNote": "Qualitative error inspection is backed by metrics/error_analysis.json, cataloging top misclassifications and borderline 0.50-probability cases."
  },
  "mlModelArtifacts": [
    {
      "name": "best_bilstm_model.pt",
      "format": "PT"
    },
    {
      "name": "vocab.json",
      "format": "JSON"
    },
    {
      "name": "baseline_tfidf_lr.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "bilstm_test_metrics.json",
      "format": "JSON"
    },
    {
      "name": "error_analysis.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Deep Learning",
    "modelArchitecture": "Bidirectional LSTM With MaxPool",
    "coreFramework": "PyTorch",
    "validationMethod": "Stratified Holdout Test Benchmark"
  }
}
```
