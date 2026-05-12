# Input layer
    input_layer = Input(shape=input_shape)

    # CNN layers with Batch Normalization
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)

    # Attention mechanism
    attention_dense = Dense(128, activation='softmax')(conv3)  # Softmax for attention weights
    attention_out = Multiply()([conv3, attention_dense])  # Element-wise multiplication

    # Global pooling
    gap = GlobalAveragePooling1D()(attention_out)

    # Dropout
    dropout = Dropout(0.4)(gap)  # Adjusted dropout rate

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dropout)

    # Model definition
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
