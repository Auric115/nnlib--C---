all:
	g++ main.cpp nnlib/NeuralNetwork.cpp nnlib/Layer.cpp nnlib/Mesh.cpp nnlib/Neuron.cpp nnlib/Connection.cpp -o test