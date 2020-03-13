import numpy
import scipy.special
from PIL import Image
import pygame


class neuralNetwork:


    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.load('Input_hidden.npy')
        self.who = numpy.load('Hidden_out.npy')

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    def query(self, inputs_list):

        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

WIDTH = 560
HEIGHT = 560
FPS = 120

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
r = 20

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("12!")
clock = pygame.time.Clock()
screen.fill(WHITE)
pygame.display.update()

run = True
while run:
    try:
        pressed = pygame.mouse.get_pressed()
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif pressed[0]:
                    pygame.draw.circle(screen, BLACK, (event.pos[0], event.pos[1]), r)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    pygame.image.save(screen, 'number.png')
                    img = Image.open('number.png').transpose(Image.TRANSPOSE)
                    img.thumbnail((28, 28))
                    pixels = img.load()
                    x = y = 28
                    image = 255 - numpy.array([pixels[i, j][0] for i in range(x) for j in range(y)])
                    inputs = (image / 255.0 * 0.99) + 0.01

                    # query the network
                    outputs = n.query(inputs)
                    label = numpy.argmax(outputs)
                    print("Ответ нейросети:", label)
                    screen.fill(WHITE)

        pygame.display.update()
    except:
        pass



