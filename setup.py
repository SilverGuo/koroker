from setuptools import setup

REQUIREMENTS = ['pip>=9.0.0',
                'tensorflow>=1.3.0, <2.0.0',
                'six>=1.11.0, <2.0.0',
                'numpy>1.13.1', ]

setup(
    name='koroker',
    version='0.0.2',
    description='python package for sequence labeling',
    url='https://github.com/silverguo/koroker',
    author='Yuhan',
    author_email='guoyuhan819@gmail.com',
    license='MIT',
    install_requires=REQUIREMENTS,
    packages=['koroker', 'koroker.prepare', 'koroker.ner',  'koroker.utils'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe=False
)
