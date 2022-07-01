// Analysis Panel

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform



ApplicationWindow {

    id: root
    width: 500 ; height: 300

    property int tb_width: 50

    ColumnLayout{
        anchors.centerIn: parent
        Text {
            Layout.alignment: Qt.AlignHCenter
            text: qsTr("Temperature Sweep Analysis")
        }

        // Temperature parameters
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Start"
                }

                IntField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    id: tstart
                    text: `${obs.TIs}`
                }

            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Step"
                }
                DoubleField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    low: 0.005
                    high: 10.0
                    prec: 4
                    text: "0.5"
                    id: tstep
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T End"
                }
                IntField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    id: tend
                    text: "10"
                }
            }
        }

        // Number of datapoints
        ColumnLayout{
            width: parent.width/2
            Text{
                Layout.alignment: Qt.AlignHCenter
                text: "Number of datapoints for every temperature"
            }

            IntField{
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                id: dpoints
                text: "12"
            }
        }


        // Waiting times
        Text{
            Layout.alignment: Qt.AlignHCenter
            text: "Waiting Times"
        }
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between datapoints"
                }

                IntField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    id: dpointwait
                    low: 1
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between Temperatures"
                }
                IntField{
                    Layout.alignment: Qt.AlignHCenter
                    id: stepwait
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Wait for equilibration"
                }
                IntField{
                    Layout.alignment: Qt.AlignHCenter
                    id: equiwait
                }
            }
        }
        ColumnLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 1

            Text{
                Layout.alignment: Qt.AlignHCenter
                text: "Save Image"
            }
            Switch{
                Layout.alignment: Qt.AlignHCenter
                checked: true
                id: saveImg
            }
        }

        Button{
            Layout.alignment: Qt.AlignHCenter
            text: "Start Analysis"
            onClicked: {
                Julia.println(`${tstart.val}`)
                Julia.tempSweepQML(tstart.val, tend.val, tstep.val, dpoints.val, dpointwait.val, stepwait.val, equiwait.val, saveImg.checked)
                close()
            }
        }
    }

}

//     function custIntValidator(text, low, high)
//     {
//         var newText = ""
//         if(isNaN(parseInt(text)))
//     {
//         var newInt = Math.max(0, low)
//         newText = `${newInt}`
//         return newText
//     }
//     if(parseInt(text) < low)
// {
//     newText = `${low}`
//     return newText
// }
// else if(parseInt(text) > high)
// {
//     newText = `${high}`
//     return newText
// }

// return text
// }


// function custDoubleValidator(text, low, high)
// {
//     var newText = ""
//     if(isNaN(parseInt(text)))
// {
//     newText = "0.0"
//     return newText
// }
// if(parseInt(text) < low)
// {
// newText = `${low}`
// return newText
// }
// else if(parseInt(text) > high)
// {
//     newText = `${high}`
//     return newText
// }

// return text
// }


