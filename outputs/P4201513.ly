\version "2.24.1"
    \header {
        title = "P4201513"
        % composer = "Yao."
    }
    \score {
        \new PianoStaff <<
            \new Staff = "right" {
                \clef treble
                \key b \major
                \time 4/4
                % \tempo 4=106
                dis'2\p fis'2\< | gis'1 | ais'4 cis'2. | ais'1 | dis'2 fis'4 dis'4 | fis'1 | dis'1\! | fis'4 b'2 dis'4 | cis'4 ais'2 cis'4 | dis'2 b'4 gis'4 | b'1 | gis'4 b'4 dis'2 | fis'2 ais'2 | b'2 gis'2 | ais'1 | gis'2. b'4 | e'4 b'4 gis'2 | cis'1 | dis'1\> | cis'2 \! \mp ais'2 | cis'2. fis'4 | ais'1 | fis'2. dis'4 | cis'4 e'4 cis'2 |
                \bar "|."
            }
            \new Staff = "left" {
                \clef bass
                \key b \major
                \time 4/4
                dis4\p fis2. | gis4\< b4 gis2 | ais2. b4 | fis4 ais4 b4 ais4 | dis4 b4 dis2 | dis4 b4 dis2 | dis4\! b4 dis2 | dis2. b4 | ais4 cis4 b4 cis4 | gis4 b2. | gis4 b2. | gis4 b2. | dis4 fis4 b4 fis4 | gis4 b2. | ais2 b4 ais4 | gis4 b2. | e4 gis2 b4 | fis4 ais2. | b4\> dis2 \! \mp b4 | fis4 ais2. | fis4 ais2 b4 | fis4 ais2. | dis4 fis2 b4 | ais4 cis2. |
                \bar "|."
            }
        >>
        \layout {}
        \midi {}
    }
    