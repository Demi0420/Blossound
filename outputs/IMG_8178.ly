\version "2.24.1"
    \header {
        title = "IMG_8178"
        % composer = "Yao."
    }
    \score {
        \new PianoStaff <<
            \new Staff = "right" {
                \clef treble
                \key g \major
                \time 9/8
                % \tempo 4=109
                g'4\p e'4. g'4. e'8\< | b'2 d'4 g'4. | d'2. 4. | c'4 a'4 fis'4. c'4 | d'4. b'2 d'4 | d'8 b'2. 4 | d'8\! fis'4 b'2 fis'4 | a'8 fis'4. a'4 fis'4. | e'4. g'4. b'4. | d'4 b'2 d'4. | c'2 8 e'4. a'8 | c'4 a'2 c'4. | g'8 e'2 g'2 | a'2 8 e'4 c'4 | c'4. e'4 c'4. g'8 | a'2. 8 fis'4 | b'2. 4. | e'4. b'2 e'4 | e'2\> g'2 \! \mp 8 | e'2. g'4. | a'4 fis'4. d'2 | d'2. 4. | b'4. g'4. d'4. | g'4 b'8 g'2. |
                \bar "|."
            }
            \new Staff = "left" {
                \clef bass
                \key g \major
                \time 9/8
                e4\p c8\< e8 c8 e4 g4 | b2 g8 b4 g4 | d2 b8 d8 g4. | a2 fis8 a8 g4. | d2 b8 d8 g4. | b2 g8 b8 g4. | d2\! b8 d8 g4. | fis2 d8 fis8 g4. | g2 e8 g2 | d2 b8 d8 g4. | c2 a8 c8 g4. | a2 fis8 a8 g4. | e2 c8 e8 g4. | c2 a8 c8 g4. | e2 c8 e8 g4. | a2 fis8 a8 g4. | b2 g8 b8 g4. | g2 e8 g2 | e2\> c8 \! \mp e8 g4. | e2 c8 e8 g4. | fis2 d8 fis8 g4. | d2 b8 d8 g4. | b2 g8 b8 g4. | g2 e8 g2 |
                \bar "|."
            }
        >>
        \layout {}
        \midi {}
    }
    