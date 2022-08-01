import src.train_networks as trainer

trainThree=trainer.trainThreeByThree(0.005,32)
trainThree.train(2)
trainThree.saveNetName("test")
trainThree.train(2)
trainThree.saveNetName("test2")
