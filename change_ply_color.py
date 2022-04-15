import colorsys

def change_color(input_path, output_path, color_tf_fn):
    output = []
    header_done = False
    with open(input_path) as f:
        for line in f:
            output_line = line
            if("end_header" in line):
                header_done = True
            if(len(line) > 40):
                if(header_done):
                    output_line = line[:find_nth(line, " ", 6)] + " " + color_tf_fn(*line.split(" ")[-5:-1]) + "\n"
            output.append(output_line)

    with open(output_path, "w") as f:
        for line in output:
            f.write(line)

def yellow_to_blue(r, g, b, a):
    # for example, a yellow duck color might be 222 178 26 255
    r = int(r)/255
    g = int(g)/255
    b = int(b)/255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb((h+0.5)%1, s, v) #do a 180 degree shift in hue
    result = str(int(r*255)) + " " + str(int(g*255)) + " " + str(int(b*255)) + " " + str(a)
    print("result", result)
    return(result)

def white(r, g, b, a):
    return "255 255 255 255"

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return(start)

if(__name__ == '__main__'):
    change_color('lm/models/obj_000009.ply', 'blue_duck.ply', yellow_to_blue)