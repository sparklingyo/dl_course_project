import os
import torch
import pickle
import argparse
from PIL import Image
import torch.nn as nn
from utils import get_cnn
from Decoder import RNN
from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from DataLoader import DataLoader, shuffle_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model')
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	args = parser.parse_args()

	with open(os.path.join(args.model, 'vocab.pkl'), 'rb') as f:
	    vocab = pickle.load(f)

	transform = transforms.Compose([transforms.Resize((224, 224)), 
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])

	embedding_dim = 512
	vocab_size = vocab.index
	print(vocab_size)
	hidden_dim = 512
	model_name = args.model
	iteration = args.epoch
	cnn = get_cnn(architecture = 'alexnet', embedding_dim = embedding_dim)
	lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
		           vocab_size = vocab_size)
	if torch.cuda.is_available():
		with torch.cuda.device(args.gpu_device):
			cnn.cuda()
			lstm.cuda()
	cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
	lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
	cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file),map_location='cuda:0'))
	lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file),map_location='cuda:0'))

		# cnn.eval()
	test_imgs_list = os.listdir('./test/')
	f = open("./test_captions/res.txt","w+")
	j = 0
	for i in test_imgs_list:
		image = transform(Image.open('./test/'+i))
		image = image.unsqueeze(0)
		
		# image = Variable(image)
		if torch.cuda.is_available():
			with torch.cuda.device(args.gpu_device):
				image = Variable(image).cuda()
		else:
			image = Variable(image)
		cnn_out = cnn(image)
		ids_list = lstm.greedy(cnn_out)
		f.write('"'+i+'"'+':'+vocab.get_sentence(ids_list)+'\n')
		if j % 400 == 0:
			print("Already captioned %d images." %(j))
		j = j+1