from setuptools import setup
from codecs import open
from os import path


from affordance_nets import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='affordance_nets',
      version=__version__,
      description='Affordance Nets: A library to compute easily the heatmaps for Robotics.',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      packages=['affordance_nets'],
      install_requires=requires_list,
      )