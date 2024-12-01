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
  --max_price 650000 \
  --min_floor_area 125 \
  --min_rooms 3 \
  --property_type apartment,house \
  --construction_period from_1945_to_1959,from_1960_to_1970,from_1971_to_1980,from_1981_to_1990,from_1991_to_2000,from_2001_to_2010,from_2011_to_2020 \
  --n_pages 100 \
  --raw_data \
  --save

open ./data