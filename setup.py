from setuptools import find_packages, setup

VERSION = '0.0.4'
DESCRIPTION = 'Probabilistic machine learning methods'
LONG_DESCRIPTION = 'A package that used for establishing multi-fidelity probabilistic machine learning models ' \
                   'Meanwhile, it also could be used for multi-fidelity Bayesian optimization and multi-fidelity' \
                   'and multi-fidelity reliability analysis'

# Setting up
setup(
    name="MfPml",
    version=VERSION,
    author="Jiaxiang Yi (Delft University of Technology), Ji Cheng (City University of Hong Kong)",
    author_email="<yagafighting@gmail.com>",
    url='https://github.com/JiaxiangYi96/MFPML.git',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    readme= "README.md",
    install_requires=['numpy', 'pandas', 'SALib', 'matplotlib', 'scipy'],
    keywords=['python', 'multi-fidelity machine learning', 'Bayesian Optimization', 'Reliability analysis'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

