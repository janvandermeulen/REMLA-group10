# Stage 3: Train the model
model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])


hist = model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )