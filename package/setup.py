from setuptools import setup

REQUIREMENTS = ['pip>=9.0.0, <10.0.0',
                'tensorflow>=1.3.0, <2.0.0',
                'six>=1.11.0, <2.0.0', ]

setup(
    name='koroker',
    version='0.0.1',
    description='python package for sequence labeling',
    url='https://gitlab.com/SilverGuo/koroker',
    author='Yuhan',
    author_email='guoyuhan819@gmail.com',
    license='MIT',
    install_requires=REQUIREMENTS,
    packages=['koroker', ],
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
