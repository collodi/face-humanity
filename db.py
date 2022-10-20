import os
import numpy as np

DB_FN = 'info.npz'

db = { 'faces': np.empty((0, 512), np.float64), 'info': np.empty((0, 3)) }

def load():
	global db
	if not os.path.exists(DB_FN):
		store_db()

	db = dict(np.load(DB_FN))

def add(embedding, user, pwd, name):
	global db
	db['faces'] = np.append(db['faces'], [embedding.detach().numpy()], axis=0)
	db['info'] = np.append(db['info'], [[user, pwd, name]], axis=0)
	store()

def store():
	global db
	np.savez(DB_FN, **db)

def find_closest_embedding(em):
	global db
	distances = [(em - e).norm().item() for e in db['faces']]
	min_d = min(distances)
	min_i = distances.index(min_d)

	return min_d, min_i

def get_user(i):
	return db['info'][i]
