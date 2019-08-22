for f in *.png; do
   n="${f/.png}"
   convert "$f" "$n.pdf"; # Comment
done