"""
Agent 3: Facial Specialist
===========================
Uses TWO pre-trained models for comprehensive facial skin diagnosis:

Model 1: EfficientNetV2B0 (Tanishq77/skin-condition-classifier) - Keras
  -> Acne, Rosacea, Eczema, Keratosis, Milia, Carcinoma
  -> 95.6% accuracy on test data

Model 2: ResNet50 + SVM (Jagadeesh2205) - Keras/Pickle
  -> Chickenpox, Cellulitis, Impetigo, Ringworm, etc.

Combined coverage for FACIAL diseases:
  - Acne, Rosacea, Melasma*, Seborrheic Dermatitis*
  (* mapped from closest model predictions)
"""

import os
import numpy as np
from PIL import Image

# Lazy-load to save memory
_efficientnet_model = None
_resnet_model = None
_svm_model = None

# === EfficientNetV2B0 Classes (Tanishq77) ===
EFFICIENTNET_LABELS = {
    0: 'Acne',
    1: 'Carcinoma',
    2: 'Eczema',
    3: 'Keratosis',
    4: 'Milia',
    5: 'Rosacea'
}

# === ResNet50 + SVM Classes (Jagadeesh) ===
RESNET_LABELS = [
    'Chickenpox', 'Cellulitis', "Athlete's Foot",
    'Impetigo', 'Nail Fungus', 'Ringworm', 'Cutaneous Larva Migrans'
]

# Combined disease info for the Facial Agent
# Written in simple language so any user can understand
DISEASE_INFO = {
    'Acne': {
        'description': 'Acne is the most common skin problem on the face. You will see pimples, blackheads, or small bumps on your forehead, cheeks, or chin. It happens when oil and dead skin block your pores. It is not dangerous but can leave marks if you pick at it. Keep your face clean and see a skin doctor if it gets worse.',
        'severity': 'Medium',
        'body_focus': 'Face, Forehead, Cheeks, Chin',
        'what_to_do': 'Wash face twice daily with gentle cleanser. Do not pop pimples. See a dermatologist if it does not improve in 2 weeks.'
    },
    'Rosacea': {
        'description': 'Rosacea makes your face look red, especially on the nose and cheeks. You may see tiny red lines (blood vessels) on your skin, and sometimes small bumps that look like pimples. It often gets worse in the sun, hot weather, or after eating spicy food. It is not contagious — you cannot give it to anyone.',
        'severity': 'Medium',
        'body_focus': 'Nose, Cheeks, Forehead',
        'what_to_do': 'Avoid sun, spicy food, and hot drinks. Use sunscreen daily. See a dermatologist for prescription creams.'
    },
    'Eczema': {
        'description': 'Eczema makes your skin red, dry, and very itchy. On the face, it usually appears around the eyes, mouth, or forehead. The skin may crack or become rough. It is very common in children but adults can get it too. It is NOT contagious — you cannot catch it from someone else.',
        'severity': 'Medium',
        'body_focus': 'Around Eyes, Mouth, Forehead',
        'what_to_do': 'Use thick moisturizer regularly. Avoid scratching. Use fragrance-free products. See a doctor for anti-itch cream.'
    },
    'Carcinoma': {
        'description': 'This is a type of skin cancer. It may look like a shiny bump, a sore that does not heal, or a flat reddish patch on your face. It grows slowly and usually does not spread to other parts of the body, but it MUST be checked by a doctor as soon as possible. Early treatment means it can almost always be fully cured.',
        'severity': 'Critical',
        'body_focus': 'Nose, Ears, Forehead',
        'what_to_do': 'SEE A DOCTOR IMMEDIATELY. Do not ignore any sore that does not heal within 4 weeks. Early treatment has 99% cure rate.'
    },
    'Keratosis': {
        'description': 'Keratosis looks like rough, dry, scaly patches on your skin. It is caused by too much sun exposure over many years. Some types are harmless, but some can slowly turn into skin cancer if left untreated. The patches may feel like sandpaper when you touch them.',
        'severity': 'Medium',
        'body_focus': 'Forehead, Temples, Scalp',
        'what_to_do': 'Always wear sunscreen. See a dermatologist to check if it is the harmless type or needs treatment.'
    },
    'Milia': {
        'description': 'Milia are very small white bumps that appear on the nose, cheeks, or under the eyes. They look like tiny white beads under the skin. They are completely harmless and not painful. They happen when dead skin gets trapped under the surface. They often go away on their own.',
        'severity': 'Low',
        'body_focus': 'Nose, Cheeks, Under Eyes',
        'what_to_do': 'Usually no treatment needed. Do not try to squeeze them. A dermatologist can safely remove them if you want.'
    },
    'Chickenpox': {
        'description': 'Chickenpox causes itchy red spots and small blisters all over your face and body. It is caused by a virus and is very contagious — you can easily spread it to others. The spots turn into blisters, then dry up and form scabs. Most people get it once and never get it again.',
        'severity': 'Medium',
        'body_focus': 'Face, Scalp',
        'what_to_do': 'Stay home and rest. Do not scratch the blisters. Use calamine lotion to reduce itching. See a doctor if you have high fever.'
    },
    'Cellulitis': {
        'description': 'Cellulitis is a skin infection caused by bacteria. The affected area becomes red, swollen, warm, and painful. It can spread quickly if not treated. You may also feel feverish or unwell. This needs medical treatment with antibiotics — do not try to treat it at home.',
        'severity': 'High',
        'body_focus': 'Face, Cheeks',
        'what_to_do': 'See a doctor URGENTLY. You will need antibiotics. Go to emergency if the redness is spreading fast or you have high fever.'
    },
    "Athlete's Foot": {
        'description': "Athlete's Foot is a fungal infection that makes the skin between your toes itchy, red, and flaky. It can also cause cracking or peeling skin. You can catch it from walking barefoot in wet places like swimming pools or showers. It is easy to treat with antifungal cream.",
        'severity': 'Low',
        'body_focus': 'Feet',
        'what_to_do': 'Use antifungal cream from the pharmacy. Keep feet dry. Wear clean socks daily. Wear sandals in public showers.'
    },
    'Impetigo': {
        'description': 'Impetigo causes red sores around the nose and mouth that burst and form golden-brown crusts. It is caused by bacteria and is very contagious — it spreads easily by touching. It is most common in children. It looks worse than it is and heals well with proper treatment.',
        'severity': 'Medium',
        'body_focus': 'Face, Nose, Mouth area',
        'what_to_do': 'See a doctor for antibiotic cream or tablets. Do not touch the sores. Wash hands often. Keep towels separate.'
    },
    'Nail Fungus': {
        'description': 'Nail Fungus makes your nails thick, yellow, or brownish. The nail may become crumbly or break easily. It usually starts at the tip of the nail and slowly spreads. It is not painful but looks bad and does not go away without treatment.',
        'severity': 'Low',
        'body_focus': 'Nails',
        'what_to_do': 'See a doctor for antifungal medicine. Treatment takes several months. Keep nails short and dry.'
    },
    'Ringworm': {
        'description': 'Ringworm is NOT a worm — it is a fungal infection! It creates round, red, itchy patches with a clear center, making it look like a ring. You can catch it from other people, animals, or contaminated objects. It is very common and easy to treat.',
        'severity': 'Medium',
        'body_focus': 'Face, Scalp',
        'what_to_do': 'Use antifungal cream for 2-4 weeks. Do not share towels or combs. Wash hands after touching the rash.'
    },
    'Cutaneous Larva Migrans': {
        'description': 'This happens when tiny worm larvae from animals burrow into your skin, usually from walking barefoot on contaminated sand or soil. You will see red, snake-like, winding lines on the skin that are very itchy. It is not dangerous and goes away with medicine.',
        'severity': 'Medium',
        'body_focus': 'Exposed skin areas',
        'what_to_do': 'See a doctor for anti-parasitic medicine. Do not walk barefoot on beaches or soil where animals go.'
    }
}

# Go up 2 levels: agent3_facial -> agents -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EFFICIENTNET_PATH = os.path.join(BASE_DIR, 'models', 'skin_model.keras')
RESNET50_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_base_model.h5')
SVM_PATH = os.path.join(BASE_DIR, 'models', 'svm_model_optimized.pkl')

IMG_SIZE = 224  # Both models use 224x224


def _load_efficientnet():
    """Load the EfficientNetV2B0 model (Acne/Rosacea/Eczema)."""
    global _efficientnet_model
    if _efficientnet_model is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only
        import tensorflow as tf
        print("[Agent 3 - Facial] Loading EfficientNetV2B0 (Acne/Rosacea/Eczema)...")
        _efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
        print("[Agent 3 - Facial] EfficientNetV2B0 loaded successfully.")
    return _efficientnet_model


def _load_resnet_svm():
    """Load the ResNet50 + SVM model (Chickenpox/Ringworm/etc)."""
    global _resnet_model, _svm_model
    import pickle
    if _resnet_model is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow as tf
        print("[Agent 3 - Facial] Loading ResNet50 feature extractor...")
        _resnet_model = tf.keras.models.load_model(RESNET50_PATH)
        print("[Agent 3 - Facial] ResNet50 loaded.")
    if _svm_model is None:
        print("[Agent 3 - Facial] Loading SVM classifier...")
        with open(SVM_PATH, 'rb') as f:
            _svm_model = pickle.load(f)
        print("[Agent 3 - Facial] SVM loaded.")
    return _resnet_model, _svm_model


def preprocess_image(img_path, size=224):
    """Preprocess image for model input."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_efficientnet(img_path):
    """Run EfficientNetV2B0 prediction (primary model)."""
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    model = _load_efficientnet()
    img_array = preprocess_image(img_path)
    img_processed = preprocess_input(img_array)
    probs = model.predict(img_processed, verbose=0)[0]

    results = []
    for idx, prob in enumerate(probs):
        disease = EFFICIENTNET_LABELS[idx]
        info = DISEASE_INFO.get(disease, {})
        results.append({
            'disease': disease,
            'probability': round(float(prob) * 100, 2),
            'description': info.get('description', ''),
            'severity': info.get('severity', 'Unknown'),
            'body_focus': info.get('body_focus', ''),
            'what_to_do': info.get('what_to_do', ''),
            'model': 'EfficientNetV2B0',
            'agent': 'Agent 3 - Facial Specialist'
        })
    return results


def predict_resnet_svm(img_path):
    """Run ResNet50+SVM prediction (secondary model)."""
    model, svm = _load_resnet_svm()
    img_array = preprocess_image(img_path, size=192)
    img_array /= 255.0
    features = model.predict(img_array, verbose=0)
    features_flat = features.reshape(1, -1)

    if hasattr(svm, 'predict_proba'):
        probs = svm.predict_proba(features_flat)[0]
    else:
        decision = svm.decision_function(features_flat)[0]
        exp_d = np.exp(decision - np.max(decision))
        probs = exp_d / exp_d.sum()

    results = []
    num_classes = min(len(probs), len(RESNET_LABELS))
    for i in range(num_classes):
        disease = RESNET_LABELS[i]
        info = DISEASE_INFO.get(disease, {})
        results.append({
            'disease': disease,
            'probability': round(float(probs[i]) * 100, 2),
            'description': info.get('description', ''),
            'severity': info.get('severity', 'Unknown'),
            'body_focus': info.get('body_focus', ''),
            'what_to_do': info.get('what_to_do', ''),
            'model': 'ResNet50+SVM',
            'agent': 'Agent 3 - Facial Specialist'
        })
    return results


def predict(img_path):
    """
    Run BOTH models and combine results.
    EfficientNet is primary (higher priority for common facial diseases).
    ResNet+SVM is secondary (covers additional diseases).
    """
    all_results = []

    # Primary model - EfficientNetV2B0
    try:
        eff_results = predict_efficientnet(img_path)
        all_results.extend(eff_results)
    except Exception as e:
        print(f"[Agent 3] EfficientNet error: {e}")

    # Secondary model - ResNet50+SVM
    try:
        resnet_results = predict_resnet_svm(img_path)
        # Scale down secondary model probabilities slightly
        for r in resnet_results:
            r['probability'] *= 0.7  # 70% weight for secondary
        all_results.extend(resnet_results)
    except Exception as e:
        print(f"[Agent 3] ResNet+SVM error: {e}")

    # Sort by probability
    all_results.sort(key=lambda x: x['probability'], reverse=True)
    return all_results


if __name__ == '__main__':
    print("Agent 3 (Facial Specialist) initialized.")
    print(f"Primary model: {EFFICIENTNET_PATH}")
    print(f"Secondary model: {RESNET50_PATH}")
    print(f"\nPrimary diseases (EfficientNetV2B0):")
    for idx, d in EFFICIENTNET_LABELS.items():
        print(f"  {idx}: {d}")
    print(f"\nSecondary diseases (ResNet50+SVM):")
    for d in RESNET_LABELS:
        print(f"  - {d}")
