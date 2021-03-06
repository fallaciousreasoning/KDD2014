def lines(match, file_name):
    """Gets all the lines from a file that contain 'match'"""
    matches = []
    with open(file_name, errors='ignore') as f:
        line = f.readline()

        index = 0
        while line is not None and line != '':
            if match in line:
                matches.append(line)

            index += 1
            line = f.readline()

    return matches

def lines_cached(match, cache):
    """Gets a line that contains 'match' from the cache and removes it"""
    if match in cache:
        #we only want one of each line, at most
        return cache.pop(match)

    return ""

def build_id_cache(lines):
    """Builds an index of id to line"""
    cache = dict()

    for line in lines:
        parts = line.split(',')
        cache.update({parts[0] : line})

    return cache


def append_matching_lines(from_file_name, match_file_name, match_column, output_file, lines_count=1000):
    """
        Appends rows from the from file to the to file where the column indexes match
    """

    done = 0
    with open(match_file_name, errors='ignore') as match_file:

        print('Caching. Trust me. You won\'t regret it')
        from_lines = []
        with open(from_file_name) as from_file:
            from_lines = build_id_cache(from_file.readlines())
        print('Done. Looking for matches')

        with open(output_file, 'w') as output_file:
            match_line = match_file.readline()
            line = 1

            while match_line is not None and match_line != '' and done < lines_count:
                parts = match_line.split(',')
                if len(parts) <= match_column:
                    match_line = match_file.readline()
                    continue

                found = lines_cached(parts[match_column], from_lines)

                if found != "":
                    done += 1
                    output_file.write(found)
                    print(done/lines_count*100, "percent done! (line", str(line) + ")")

                match_line = match_file.readline()
                line += 1

    print('done! (told you)')


append_matching_lines('projects.csv', 'donations.csv', 1, 'projects_small.csv')