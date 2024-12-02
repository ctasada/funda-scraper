#!/bin/bash

# Check if the area argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <area>"
  exit 1
fi

# Assign the first argument to the area variable
AREA=$1

# Execute the command with the dynamic area
poetry run python funda_scraper/scrape.py \
  --area "$AREA" \
  --want_to buy \
  --max_price 800000 \
  --min_floor_area 125 \
  --min_rooms 3 \
  --property_type apartment \
  --construction_period from_1945_to_1959,from_1960_to_1970,from_1971_to_1980,from_1981_to_1990,from_1991_to_2000,from_2001_to_2010,from_2011_to_2020 \
  --n_pages 100 \
  --raw_data \
  --save

open ./data



https://www.funda.nl/en/zoeken/koop?selected_area=%5B%22amsterdam%22%5D&object_type=%5B%22apartment%22%5D&availability=%5B"available"%5D&price=%22-800000%22&floor_area=%22125-%22&rooms=%223-%22&construction_period=%5B%22from_1945_to_1959%22,%22from_1960_to_1970%22,%22from_1971_to_1980%22,%22from_1981_to_1990%22,%22from_1991_to_2000%22,%22from_2001_to_2010%22,%22from_2011_to_2020%22%5D
https://www.funda.nl/en/zoeken/koop?selected_area=%5B%22amsterdam%22%5D&object_type=%5B%22apartment%22%5D&availability=%5B%22available%22%5D&floor_area=%22125-%22&rooms=%223-%22&construction_period=%5B%22from_1945_to_1959%22,%22from_1960_to_1970%22,%22from_1971_to_1980%22,%22from_1981_to_1990%22,%22from_1991_to_2000%22,%22from_2001_to_2010%22,%22from_2011_to_2020%22%5D