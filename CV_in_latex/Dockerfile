FROM texlive/texlive:latest

# Install required LaTeX packages
RUN tlmgr update --self && \
    tlmgr install \
    moderncv \
    ulem \
    xcolor \
    geometry \
    cjk \
    fontawesome5 \
    simpleicons \
    lm \
    helvetic

# Install additional utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /latex

# Copy LaTeX files
COPY main.tex /latex/
COPY avatar.jpg /latex/

# Command to build the CV
CMD ["pdflatex", "main.tex"]
