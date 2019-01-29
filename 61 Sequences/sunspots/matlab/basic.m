load '/home/peter/Documents/Studie Jaar 8/thesis/datasets/sunspots/raw/andrews14.dat' 

arot = andrews14';
sunspots = arot(:);

csvwrite('sunspots.csv', sunspots);
