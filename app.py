import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

import time

# Optimized Configuration
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 16
EPOCHS = 25  # Strictly 25 epochs for initial training
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 0.0001

# Check GPU availability
print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
if tf.config.list_physical_devices('GPU'):
    print("GPU Names:", tf.config.list_physical_devices('GPU'))
    print("GPU will be used for training!")
else:
    print("No GPU detected - training on CPU")
print("=" * 60)

# Disease information
DISEASE_INFO = {
    'acne': {
        'name': 'Acne',
        'diagnosis': 'Acne is a common skin condition caused by clogged hair follicles. Treatment includes topical retinoids, benzoyl peroxide, or antibiotics. Maintain good skincare routine and avoid picking.',
        'severity': 'Mild to Moderate'
    },
    'eczema': {
        'name': 'Eczema (Atopic Dermatitis)',
        'diagnosis': 'Eczema is a chronic inflammatory skin condition. Use gentle moisturizers, avoid triggers, and consider topical corticosteroids or immunomodulators as prescribed by a dermatologist.',
        'severity': 'Mild to Severe'
    },
    'melanoma': {
        'name': 'Melanoma',
        'diagnosis': 'Melanoma is a serious form of skin cancer. IMMEDIATE medical attention required. Early detection and surgical removal are crucial. Regular dermatological checkups recommended.',
        'severity': 'Severe - Requires Immediate Medical Attention'
    },
    'psoriasis': {
        'name': 'Psoriasis',
        'diagnosis': 'Psoriasis is an autoimmune condition causing rapid skin cell turnover. Treatment may include topical treatments, phototherapy, or systemic medications. Consult a dermatologist.',
        'severity': 'Moderate to Severe'
    },
    'rosacea': {
        'name': 'Rosacea',
        'diagnosis': 'Rosacea causes facial redness and inflammation. Avoid triggers like sun exposure, alcohol, and spicy foods. Use gentle skincare and consider topical medications.',
        'severity': 'Mild to Moderate'
    },
    'dermatitis': {
        'name': 'Contact Dermatitis',
        'diagnosis': 'Contact dermatitis is caused by allergens or irritants. Identify and avoid triggers, use gentle cleansers, and apply topical corticosteroids as needed.',
        'severity': 'Mild to Moderate'
    },
    'basal_cell_carcinoma': {
        'name': 'Basal Cell Carcinoma',
        'diagnosis': 'Basal cell carcinoma is the most common skin cancer. Medical evaluation and treatment required. Usually treatable with surgery, radiation, or topical treatments.',
        'severity': 'Moderate - Requires Medical Treatment'
    }
}

def create_data_generators(train_dir, validation_dir, test_dir):
    """Create data generators with optimized augmentation for your dataset structure"""
    
    # Enhanced augmentation for training
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,           
    width_shift_range=0.25,      
    height_shift_range=0.25,
    shear_range=0.25,           
    zoom_range=0.25,            
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3], 
    fill_mode='nearest',
    channel_shift_range=20.0,    # Color variation
    preprocessing_function=lambda x: x + np.random.normal(0, 0.01, x.shape)  # Gaussian noise
)

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Define class names matching your dataset folders
    class_names = ['Acne', 'Basal cell Carcinoma', 'Contact Dermatitis', 
                   'Eczema', 'Melanoma', 'Psoriasis', 'Rosacea']
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=True,
        seed=42
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def load_base_model_with_retry(model_func, max_retries=3):
    """Load base model with retry logic for network issues"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries} to load model weights...")
            model = model_func()
            print(" Model weights loaded successfully!")
            return model
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("\n Failed to load model after all retries.")
                print("Please check your internet connection and try again.")
                raise

def create_resnet50_model(num_classes):
    """Create optimized ResNet50 model"""
    
    def load_resnet():
        return ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    
    base_model = load_base_model_with_retry(load_resnet)
    
    # Store base model reference for fine-tuning
    base_model.trainable = False
    
    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(x)
    x = Dropout(0.4, name='dropout_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='ResNet50_SkinDisease')
    
    # Store base model as an attribute for easy access
    model.base_model = base_model
    
    return model

def create_efficientnet_model(num_classes):
    """Create optimized EfficientNetB4 model"""
    
    def load_efficientnet():
        return EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    
    base_model = load_base_model_with_retry(load_efficientnet)
    
    # Store base model reference for fine-tuning
    base_model.trainable = False
    
    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(x)
    x = Dropout(0.4, name='dropout_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='EfficientNetB4_SkinDisease')
    
    # Store base model as an attribute for easy access
    model.base_model = base_model
    
    return model

def calculate_class_weights(train_generator):
    """Calculate class weights to handle imbalanced data"""
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    return dict(enumerate(class_weights))

def lr_schedule(epoch, lr):
    """Learning rate schedule with warmup and decay"""
    if epoch < 3:
        # Warmup phase
        return LEARNING_RATE * (epoch + 1) / 3
    elif epoch < 15:
        return LEARNING_RATE
    elif epoch < 20:
        return LEARNING_RATE * 0.5
    else:
        return LEARNING_RATE * 0.1

def train_model(model, train_generator, validation_generator, model_name, class_weights_dict):
    """Train the model with optimized callbacks - STRICTLY 25 EPOCHS"""
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}_initial_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            #save_format='keras'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting Initial Training for {model_name}")
    print(f"EPOCHS: {EPOCHS} (STRICTLY ENFORCED)")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,  # Strictly 25 epochs
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    return model, history

def fine_tune_model(model, train_generator, validation_generator, model_name, class_weights_dict, layers_to_unfreeze=30):
    """Fine-tune model by unfreezing layers - FIXED BASE MODEL EXTRACTION"""
    
    # Access the stored base model
    if not hasattr(model, 'base_model'):
        print(f"⚠️ Warning: Could not find base model attribute for fine-tuning {model_name}")
        return model, None
    
    base_model = model.base_model
    base_model.trainable = True
    
    total_layers = len(base_model.layers)
    
    # Unfreeze only the last N layers
    for i, layer in enumerate(base_model.layers):
        if i < total_layers - layers_to_unfreeze:
            layer.trainable = False
        else:
            layer.trainable = True
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Configuration for {model_name}")
    print(f"{'='*60}")
    print(f"Total layers in base model: {total_layers}")
    print(f"Unfreezing last {layers_to_unfreeze} layers")
    print(f"Trainable layers: {trainable_count}")
    print(f"Frozen layers: {total_layers - trainable_count}")
    print(f"{'='*60}\n")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}_finetuned_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            #save_format='keras'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting Fine-Tuning for {model_name}")
    print(f"Fine-tune epochs: {FINE_TUNE_EPOCHS}")
    print(f"{'='*60}\n")
    
    history_fine = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    return model, history_fine

def plot_training_history(history, history_fine, model_name):
    """Plot combined training history"""
    
    os.makedirs('plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    initial_epochs = len(acc)
    
    if history_fine:
        acc.extend(history_fine.history['accuracy'])
        val_acc.extend(history_fine.history['val_accuracy'])
        loss.extend(history_fine.history['loss'])
        val_loss.extend(history_fine.history['val_loss'])
    
    epochs_range = range(len(acc))
    
    # Accuracy plot
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    if history_fine:
        ax1.axvline(x=initial_epochs-0.5, color='g', linestyle='--', linewidth=2, label='Fine-tuning Start')
    ax1.set_title(f'{model_name} - Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Loss plot
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    if history_fine:
        ax2.axvline(x=initial_epochs-0.5, color='g', linestyle='--', linewidth=2, label='Fine-tuning Start')
    ax2.set_title(f'{model_name} - Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: plots/{model_name}_training_history.png")

def evaluate_model(model, test_generator, class_names, model_name):
    """Evaluate model on test set with detailed metrics"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}")
    
    # Reset test generator
    test_generator.reset()
    
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    test_loss, test_accuracy, test_top2 = model.evaluate(test_generator, verbose=0)
    print(f"\n{'='*60}")
    print(f"Test Results for {model_name}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Top-2 Accuracy: {test_top2*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'='*60}\n")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, square=True,
                annot_kws={'size': 10})
    plt.title(f'{model_name} - Confusion Matrix (Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: plots/{model_name}_confusion_matrix.png")
    
    print("\n" + "="*60)
    print("Per-Class Accuracy Analysis")
    print("="*60)
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        accuracy = (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0
        print(f"{class_name:25s}: {accuracy:6.2f}% ({class_correct[i]:3d}/{class_total[i]:3d} correct)")
    print("="*60)
    
    return test_accuracy

def main():
    """Main training function"""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Dataset paths
    train_dir = 'dataset/training'
    validation_dir = 'dataset/validation'
    test_dir = 'dataset/testing'
    
    # Verify dataset structure
    print("\n" + "="*60)
    print("VERIFYING DATASET STRUCTURE")
    print("="*60)
    for dir_name, dir_path in [('Training', train_dir), ('Validation', validation_dir), ('Testing', test_dir)]:
        if os.path.exists(dir_path):
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            print(f"✓ {dir_name:12s}: {len(subdirs)} classes found")
        else:
            print(f"✗ {dir_name:12s}: Directory not found at {dir_path}")
    print("="*60)
    
    print("\nCreating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, validation_dir, test_dir
    )
    
    num_classes = 7
    class_names = list(train_generator.class_indices.keys())
    
    print(f"\n{'='*60}")
    print("DATASET INFORMATION")
    print(f"{'='*60}")
    print(f"Training samples:   {train_generator.samples:5d} ({train_generator.samples//num_classes} per class)")
    print(f"Validation samples: {validation_generator.samples:5d} ({validation_generator.samples//num_classes} per class)")
    print(f"Test samples:       {test_generator.samples:5d} ({test_generator.samples//num_classes} per class)")
    print(f"Number of classes:  {num_classes}")
    print(f"Class names:        {class_names}")
    print(f"\nTraining Configuration:")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Initial epochs:   {EPOCHS} (STRICTLY ENFORCED)")
    print(f"  Fine-tune epochs: {FINE_TUNE_EPOCHS}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Image size:       {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"{'='*60}")
    
    print("\nCalculating class weights to handle class imbalance...")
    class_weights_dict = calculate_class_weights(train_generator)
    print("\nClass weights:")
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name:25s}: {class_weights_dict[cls_idx]:.4f}")
    
    results = {}
    
    # Train ResNet50
    print("\n" + "="*60)
    print("TRAINING RESNET50 MODEL")
    print("="*60)
    
    try:
        resnet50_model = create_resnet50_model(num_classes)
        print(f"\nModel created: {resnet50_model.name}")
        print(f"Total parameters: {resnet50_model.count_params():,}")
        
        resnet50_model, resnet50_history = train_model(
            resnet50_model, train_generator, validation_generator, 
            "ResNet50", class_weights_dict
        )
        
        print("\n" + "="*60)
        print("FINE-TUNING RESNET50")
        print("="*60)
        resnet50_model, resnet50_fine_history = fine_tune_model(
            resnet50_model, train_generator, validation_generator, 
            "ResNet50", class_weights_dict, layers_to_unfreeze=30
        )
        
        # Save final model
        resnet50_model.save("models/resnet50_skin_disease_final.keras")
        print(f"✓ Saved: models/resnet50_skin_disease_final.keras")
        
        plot_training_history(resnet50_history, resnet50_fine_history, "ResNet50")
        resnet50_accuracy = evaluate_model(resnet50_model, test_generator, class_names, "ResNet50")
        results['ResNet50'] = resnet50_accuracy
        
        print("\n✓ ResNet50 training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error training ResNet50: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Continuing with EfficientNetB4...")
    
    # Train EfficientNetB4
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNETB4 MODEL")
    print("="*60)
    
    try:
        efficientnet_model = create_efficientnet_model(num_classes)
        print(f"\nModel created: {efficientnet_model.name}")
        print(f"Total parameters: {efficientnet_model.count_params():,}")
        
        efficientnet_model, efficientnet_history = train_model(
            efficientnet_model, train_generator, validation_generator, 
            "EfficientNetB4", class_weights_dict
        )
        
        print("\n" + "="*60)
        print("FINE-TUNING EFFICIENTNETB4")
        print("="*60)
        efficientnet_model, efficientnet_fine_history = fine_tune_model(
            efficientnet_model, train_generator, validation_generator, 
            "EfficientNetB4", class_weights_dict, layers_to_unfreeze=50
        )
        
        # Save final model
        efficientnet_model.save("models/efficientnet_skin_disease_final.keras")
        print(f" Saved: models/efficientnet_skin_disease_final.keras")
        
        plot_training_history(efficientnet_history, efficientnet_fine_history, "EfficientNetB4")
        efficientnet_accuracy = evaluate_model(efficientnet_model, test_generator, class_names, "EfficientNetB4")
        results['EfficientNetB4'] = efficientnet_accuracy
        
        print("\n EfficientNetB4 training completed successfully!")
        
    except Exception as e:
        print(f"\n Error training EfficientNetB4: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save class names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"\n Saved: class_names.json")
    
    # Final Results
    if results:
        print("\n" + "="*60)
        print("FINAL RESULTS - Test Accuracy Comparison")
        print("="*60)
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:20s}: {accuracy*100:.2f}%")
        
        best_model = max(results, key=results.get)
        print(f"\n🏆 BEST MODEL: {best_model} with {results[best_model]*100:.2f}% test accuracy")
        print("="*60)
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - models/ResNet50_initial_best.keras")
    print("  - models/ResNet50_finetuned_best.keras")
    print("  - models/resnet50_skin_disease_final.keras")
    print("  - models/EfficientNetB4_initial_best.keras")
    print("  - models/EfficientNetB4_finetuned_best.keras")
    print("  - models/efficientnet_skin_disease_final.keras")
    print("  - plots/ResNet50_training_history.png")
    print("  - plots/ResNet50_confusion_matrix.png")
    print("  - plots/EfficientNetB4_training_history.png")
    print("  - plots/EfficientNetB4_confusion_matrix.png")
    print("  - class_names.json")
    print("="*60)

if __name__ == "__main__":
    main()