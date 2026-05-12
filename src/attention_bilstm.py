#attention
query = Dense(inputs.shape[-1])(inputs)
    attention_scores = Permute([2, 1])(query)
    attention_weights = Dense(1, activation='softmax')(attention_scores)
    attention_weights = Permute([2, 1])(attention_weights)
    weighted_inputs = Multiply()([inputs, attention_weights])
    return Lambda(lambda x: K.sum(x, axis=1))(weighted_inputs)
#bilstm
inputs = Input(shape=input_shape)
    bilstm = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    attention = attention_layer(bilstm)
    dropout = Dropout(0.3)(attention)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
