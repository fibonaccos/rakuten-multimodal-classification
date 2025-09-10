from cnn import CNNModel


cnn = CNNModel(42)

cnn.load_datasets()
cnn.build()

cnn.summary()


history = cnn.train()
metrics = cnn.metrics_callback_.history
