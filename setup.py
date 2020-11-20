from setuptools import find_packages, setup


with open('requirements.txt', 'r') as file:
    install_requires = [name.strip() for name in file]


with open('requirements-visual.txt', 'r') as file:
    full_requires = [name.strip() for name in file]


setup(
    name='rl-starterpack',
    version='0.1',
    description='RL Starterpack',
    long_description='RL Starterpack',
    author='Alexander Kuhnle',
    author_email='alexkuhnle@t-online.de',
    url='',
    packages=['rl_starterpack'],
    download_url='',
    license='',
    python_requires='>=3.5',
    install_requires=install_requires,
    extras_require=dict(full=full_requires)
)
