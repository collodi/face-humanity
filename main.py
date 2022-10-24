import cv2
import torch
import numpy as np
import PySimpleGUI as sg
from facenet_pytorch import MTCNN, InceptionResnetV1

import db
import humanity

mtcnn = MTCNN().eval()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def main():
	db.load()

	layout = [
			[sg.Button('Log In')],
			[sg.Button('Register A New Face')],
			[sg.Button('Quit')]
	]

	window = sg.Window('Humanity Face', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED or event == 'Quit':
			break
		elif event == 'Log In':
			login()
		elif event == 'Register A New Face':
			register()

	window.close()

def logged_in(user_ind):
	user, pwd, name = db.get_user(user_ind)

	layout = [
			[sg.Text(f'Hi, {name}!')],
			[sg.Button('Clock In')],
			[sg.Button('Clock Out')],
			[sg.Button('Log Out')]
	]

	window = sg.Window('Humanity Face', layout, modal=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED or event == 'Log Out':
			break
		elif event == 'Clock In':
			success = humanity.clock_in()
			if success: # TODO
				pass
			else:
				pass
		elif event == 'Clock Out':
			success = humanity.clock_out()
			if success: # TODO
				pass
			else:
				pass

	window.close()

def login():
	camera = cv2.VideoCapture(0)

	frame = capture_camera(camera)
	zeros = np.zeros((600, 360))
	if frame is not None:
		zeros = np.zeros(frame.shape)

	imgbytes = cv2.imencode('.png', zeros)[1].tobytes()

	layout = [
			[sg.Image(data=imgbytes, key='image')],
			[sg.Button('Go Back')]
	]

	window = sg.Window('Log In', layout, modal=True)

	action_timeout = 0
	while True:
		event, values = window.read(timeout=5)
		if event == sg.WINDOW_CLOSED or event == 'Go Back':
			break

		frame = capture_camera(camera)
		boxes, probs = mtcnn.detect(frame)

		img = frame
		if boxes is not None and probs[0] > 0.5:
			face = mtcnn.extract(frame[..., ::-1], boxes, None)
			embedding = resnet(face.unsqueeze(0))[0].detach()

			l, t, r, b = boxes[0].astype(int)
			img = cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)

			dist, ind = db.find_closest_embedding(embedding)
			if dist < 0.65:
				camera.release()
				window.close()
				logged_in(ind)
				return

		imgbytes = cv2.imencode('.png', img)[1].tobytes()
		window['image'].update(data=imgbytes)

		# timeout (close window) to save resources
		action_timeout += 1
		if action_timeout == 300:
			break

	camera.release()
	window.close()

def show_msg(msg):
	layout = [
			[sg.Text(msg)],
			[sg.Button('Okay')]
	]

	window = sg.Window('Message', layout, modal = True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED or event == 'Okay':
			break

	window.close()

def tensor_to_pngbytes(tensor):
	arr = np.rollaxis(tensor.numpy(), 0, 3)[..., ::-1]
	arr = ((arr + 1) * 128).astype(np.int16)
	return cv2.imencode('.png', arr)[1].tobytes()

def register():
	face = capture_face()
	if face is None:
		show_msg('Could not find a face! Please try again.')
		return
	elif face is False:
		return

	layout = [
			[sg.Image(data=tensor_to_pngbytes(face), key='image')],
			[sg.Text('Username', size=(10, 1)), sg.InputText(key='username')],
			[sg.Text('Password', size=(10, 1)), sg.InputText(key='password')],
			[sg.Button('Submit'), sg.Button('Go Home')],
	]

	window = sg.Window('Register A New Face', layout, modal=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED or event == 'Go Home':
			break
		elif event == 'Submit':
			user = values['username']
			pwd = values['password']

			name = humanity.get_name(user, pwd)
			if name is not None:
				embedding = resnet(face.unsqueeze(0))[0]
				db.add(embedding, user, pwd, name)
				break
			else:
				show_msg('Could not verify the user. Please try again.')

	window.close()

def capture_face():
	camera = cv2.VideoCapture(0)
	face = None

	frame = capture_camera(camera)
	zeros = np.zeros((600, 360))
	if frame is not None:
		zeros = np.zeros(frame.shape)

	imgbytes = cv2.imencode('.png', zeros)[1].tobytes()

	layout = [
			[sg.Image(data=imgbytes, key='image')],
			[sg.Button('Capture Face'), sg.Button('Go Back')]
	]

	window = sg.Window('Capture Face', layout, modal=True)

	# TODO only detect & update face once in so many loops
	while True:
		event, values = window.read(timeout=20)
		if event == sg.WINDOW_CLOSED or event == 'Go Back':
			camera.release()
			window.close()
			return False

		frame = capture_camera(camera)
		boxes, probs = mtcnn.detect(frame)

		img = frame
		if boxes is not None and probs[0] > 0.5:
			l, t, r, b = boxes[0].astype(int)
			face = mtcnn.extract(frame[..., ::-1], boxes, None)
			img = cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)

		imgbytes = cv2.imencode('.png', img)[1].tobytes()
		window['image'].update(data=imgbytes)

		if event == 'Capture Face':
			break

	camera.release()
	window.close()
	return face

def capture_camera(camera):
	ret, frame = camera.read()
	if frame is None:
		return None

	h, w, d =  frame.shape
	x_s = max(0, w // 2 - 180)
	x_e = w // 2 + 180

	return frame[:, x_s:x_e, :]

if __name__ == '__main__':
	main()
